import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
import itertools

# Importaciones del código base del profesor
from Environments.AbstractEnv import AbstractEnv
from Environments.PartiallyObservableEnvs.InvisibleDoorEnv import InvisibleDoorEnv
from MemoryWrappers.KOrderMemory import KOrderMemory
from MemoryWrappers.BinaryMemory import BinaryMemory

# --- Implementación de la Nueva Memoria (Pregunta g) ---
class SaveableKOrderMemory(AbstractEnv):
    """
    Una k-order memory donde el agente decide si guardar o no la observación actual.
    """
    def __init__(self, env, memory_size):
        self.__env = env
        self.__memory_size = memory_size
        self.__memory = deque(maxlen=self.__memory_size)
        self.__env_state = None

        # El espacio de acciones se expande
        self.__env_actions = self.__env.action_space
        self.__memory_actions = ["save", "ignore"]
        self.__action_space = list(itertools.product(self.__env_actions, self.__memory_actions))

    @property
    def action_space(self):
        return self.__action_space

    def reset(self):
        self.__env_state = self.__env.reset()
        self.__memory.clear()
        return self.__get_state()

    def __get_state(self):
        # El estado es la observación actual concatenada con la memoria
        return (self.__env_state,) + tuple(self.__memory)

    def step(self, action):
        env_action, mem_action = action
        
        # Guardar la observación ANTES de ejecutar la acción si se indica
        if mem_action == "save":
            self.__memory.append(self.__env_state)

        # Ejecutar la acción en el entorno
        self.__env_state, r, done = self.__env.step(env_action)
        
        return self.__get_state(), r, done

    def show(self):
        self.__env.show()
        print(f"Buffer: {list(self.__memory)}")


# --- Función Genérica para los Experimentos de RL ---
def run_rl_experiment(env, num_episodes, alpha, gamma, epsilon, n_step=1, algorithm='q_learning'):
    q_table = defaultdict(lambda: 1.0)
    episode_lengths = []

    for episode_num in range(num_episodes):
        state = env.reset()
        
        actions_history = []
        states_history = [state]
        rewards_history = [0]

        steps = 0
        done = False
        while not done:
            current_state = states_history[steps]
            
            if np.random.rand() < epsilon:
                action = random.choice(env.action_space)
            else:
                action = max(env.action_space, key=lambda a: q_table[(current_state, a)])
            actions_history.append(action)

            next_state, reward, done = env.step(action)
            states_history.append(next_state)
            rewards_history.append(reward)
            
            tau = steps - n_step + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(steps + 2, tau + n_step + 1)):
                    G += (gamma**(i - tau - 1)) * rewards_history[i]
                
                if tau + n_step <= steps:
                    bootstrap_state = states_history[tau + n_step]
                    if algorithm == 'q_learning':
                        q_next = max(q_table[(bootstrap_state, a)] for a in env.action_space)
                    else: # Sarsa
                        action_next = actions_history[tau + n_step]
                        q_next = q_table[(bootstrap_state, action_next)]
                    G += (gamma**n_step) * q_next

                update_state = states_history[tau]
                update_action = actions_history[tau]
                
                current_q = q_table[(update_state, update_action)]
                q_table[(update_state, update_action)] = current_q + alpha * (G - current_q)

            steps += 1
            if steps >= 5000: # Límite de tiempo
                done = True

        episode_lengths.append(steps)
    return episode_lengths

def plot_results(results, title_suffix):
    plt.figure(figsize=(14, 8))
    for label, lengths_runs in results.items():
        mean_lengths = np.mean(lengths_runs, axis=0)
        plt.plot(mean_lengths, label=label)
        
    plt.xlabel('Episodios')
    plt.ylabel('Largo Promedio de Episodio')
    plt.title(f'Rendimiento en InvisibleDoorEnv ({title_suffix})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'invisible_door_{title_suffix.lower().replace(" ", "_")}.png')
    plt.show()

# --- Script Principal ---
if __name__ == '__main__':
    num_runs = 30
    num_episodes = 1000
    gamma = 0.99
    alpha = 0.1
    epsilon = 0.01

    algorithms_to_run = {
        'Q-Learning': {'algorithm': 'q_learning', 'n_step': 1},
        'Sarsa': {'algorithm': 'sarsa', 'n_step': 1},
        '16-step Sarsa': {'algorithm': 'sarsa', 'n_step': 16},
    }

    '''
    # --- 1. Experimento SIN Memoria ---
    print("--- Ejecutando SIN Memoria ---")
    results_no_memory = {}
    env_base = InvisibleDoorEnv()
    for name, params in algorithms_to_run.items():
        print(f"Ejecutando {name}...")
        runs = [run_rl_experiment(env_base, num_episodes, alpha, gamma, epsilon, **params) for _ in range(num_runs)]
        results_no_memory[name] = runs
    plot_results(results_no_memory, "Sin Memoria")

    # --- 2. Experimento con K-Order Memory ---
    print("\n--- Ejecutando con 2-Order Memory ---")
    results_k_order = {}
    env_k_order = KOrderMemory(InvisibleDoorEnv(), memory_size=2)
    for name, params in algorithms_to_run.items():
        print(f"Ejecutando {name}...")
        runs = [run_rl_experiment(env_k_order, num_episodes, alpha, gamma, epsilon, **params) for _ in range(num_runs)]
        results_k_order[name] = runs
    plot_results(results_k_order, "2-Order Memory")

    # --- 3. Experimento con Binary Memory ---
    print("\n--- Ejecutando con Binary Memory ---")
    results_binary = {}
    env_binary = BinaryMemory(InvisibleDoorEnv(), num_of_bits=1)
    for name, params in algorithms_to_run.items():
        print(f"Ejecutando {name}...")
        runs = [run_rl_experiment(env_binary, num_episodes, alpha, gamma, epsilon, **params) for _ in range(num_runs)]
        results_binary[name] = runs
    plot_results(results_binary, "Binary Memory (1 bit)")
'''
    # --- 4. Experimento con Nueva Memoria ---
    print("\n--- Ejecutando con Saveable K-Order Memory ---")
    results_saveable = {}
    env_saveable = SaveableKOrderMemory(InvisibleDoorEnv(), memory_size=1)
    for name, params in algorithms_to_run.items():
        print(f"Ejecutando {name}...")
        runs = [run_rl_experiment(env_saveable, num_episodes, alpha, gamma, epsilon, **params) for _ in range(num_runs)]
        results_saveable[name] = runs
    plot_results(results_saveable, "Saveable Memory (1 slot)")