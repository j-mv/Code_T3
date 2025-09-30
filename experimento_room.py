import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from Environments.MultiGoalEnvs.RoomEnv import RoomEnv

def run_experiment(env, num_episodes, alpha, gamma, epsilon, n_step=1, is_multi_goal=False, algorithm='q_learning'):
    q_table = defaultdict(lambda: 1.0)
    all_goals = env.goals
    episode_lengths = []

    for _ in range(num_episodes):
        # Iniciar episodio
        state = env.reset()
        
        # Guardar la trayectoria: S_0, A_0, R_1, S_1, A_1, R_2, ...
        states = [state]
        actions = []
        rewards = [0]  # R_0 es 0 por convención

        steps = 0
        done = False
        while not done:
            # Estado actual es S_t (t=steps)
            current_pos, current_goal = states[steps]
            current_q_state = (current_pos, current_goal)

            # Seleccionar acción A_t
            if np.random.rand() < epsilon:
                action = random.choice(env.action_space)
            else:
                action = max(env.action_space, key=lambda a: q_table.get((current_q_state, a), 1.0))
            actions.append(action)

            # Ejecutar A_t, observar R_{t+1} y S_{t+1}
            next_state, reward, done = env.step(action)
            states.append(next_state)
            rewards.append(reward)
            
            # --- Actualización N-Step ---
            # τ es el paso que vamos a actualizar. Podemos hacerlo ahora porque tenemos info hasta R_{τ+n}
            tau = steps - n_step + 1
            if tau >= 0:
                G = 0
                # 1. Calcular suma de recompensas
                for i in range(tau + 1, min(steps + 2, tau + n_step + 1)):
                    G += (gamma**(i - tau - 1)) * rewards[i]
                
                # 2. Añadir bootstrap si no es terminal
                if tau + n_step <= steps:
                    bootstrap_pos, bootstrap_goal = states[tau + n_step]
                    bootstrap_q_state = (bootstrap_pos, bootstrap_goal)
                    
                    if algorithm == 'q_learning':
                        q_next = max(q_table.get((bootstrap_q_state, a), 1.0) for a in env.action_space)
                    else: # Sarsa
                        action_next = actions[tau + n_step]
                        q_next = q_table.get((bootstrap_q_state, action_next), 1.0)
                    G += (gamma**n_step) * q_next

                # 3. Actualizar Q(S_τ, A_τ)
                update_pos, update_goal = states[tau]
                update_q_state = (update_pos, update_goal)
                update_action = actions[tau]
                
                if not is_multi_goal:
                    q_table[(update_q_state, update_action)] += alpha * (G - q_table.get((update_q_state, update_action), 1.0))
                else: # Lógica Multi-Goal (para n=1)
                    s_pos, s_goal = states[tau]
                    s_next_pos, _ = states[tau+1]
                    
                    for g_prime in all_goals:
                        hypothetical_state = (s_pos, g_prime)
                        r_prime = 1.0 if s_next_pos == g_prime else 0.0
                        G_prime = r_prime

                        if s_next_pos != g_prime:
                            hypothetical_next_state = (s_next_pos, g_prime)
                            if algorithm == 'q_learning':
                                q_next_prime = max(q_table.get((hypothetical_next_state, a), 1.0) for a in env.action_space)
                            else: # Sarsa
                                # **LA CORRECCIÓN FINAL Y CRÍTICA ESTÁ AQUÍ**
                                # Nos aseguramos de que la acción exista.
                                # A_{tau+1} es la acción para S_{tau+1}
                                if tau + 1 < len(actions):
                                    next_action_prime = actions[tau + 1]
                                    q_next_prime = q_table.get((hypothetical_next_state, next_action_prime), 1.0)
                                else:
                                    q_next_prime = 0 # El episodio terminó, no hay acción siguiente
                            G_prime += gamma * q_next_prime
                        
                        q_table[(hypothetical_state, update_action)] += alpha * (G_prime - q_table.get((hypothetical_state, update_action), 1.0))

            steps += 1
        episode_lengths.append(steps)
    return episode_lengths

def plot_results(results, title):
    plt.figure(figsize=(14, 8))
    for label, lengths in results.items():
        mean_lengths = np.mean(lengths, axis=0)
        plt.plot(mean_lengths, label=label)
    plt.xlabel('Episodios')
    plt.ylabel('Largo Promedio de Episodio')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('room_env_results.png')
    plt.show()

if __name__ == '__main__':
    num_runs = 100
    num_episodes = 500
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.99
    
    env = RoomEnv()
    all_results = {}
    
    algorithms_to_run = {
        'Q-Learning': {'algorithm': 'q_learning', 'n_step': 1, 'is_multi_goal': False},
        'Sarsa': {'algorithm': 'sarsa', 'n_step': 1, 'is_multi_goal': False},
        '8-step Sarsa': {'algorithm': 'sarsa', 'n_step': 8, 'is_multi_goal': False},
        'Multi-Goal Q-Learning': {'algorithm': 'q_learning', 'n_step': 1, 'is_multi_goal': True},
        'Multi-Goal Sarsa': {'algorithm': 'sarsa', 'n_step': 1, 'is_multi_goal': True},
    }

    for name, params in algorithms_to_run.items():
        print(f"Ejecutando {name}...")
        runs_lengths = []
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}", end='\r')
            env.reset()
            lengths = run_experiment(env, num_episodes, alpha, gamma, epsilon, **params)
            runs_lengths.append(lengths)
        all_results[name] = runs_lengths
        print(f"\n¡{name} completado!")

    plot_results(all_results, 'Rendimiento en RoomEnv')