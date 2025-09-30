import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from Environments.MultiAgentEnvs.CentralizedHunterEnv import CentralizedHunterEnv
from Environments.MultiAgentEnvs.HunterEnv import HunterEnv
from Environments.MultiAgentEnvs.HunterAndPreyEnv import HunterAndPreyEnv

def q_learning_centralized(env, num_episodes, alpha, gamma, epsilon):
    """ Un solo agente aprende una política sobre acciones conjuntas. """
    q_table = defaultdict(lambda: {action: 1.0 for action in env.action_space})
    episode_lengths = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            steps += 1
            if np.random.rand() < epsilon:
                action = random.choice(env.action_space)
            else:
                action = max(q_table[state], key=q_table[state].get)
            
            next_state, reward, done = env.step(action)
            
            old_value = q_table[state][action]
            next_max = max(q_table[next_state].values())
            
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state][action] = new_value
            
            state = next_state
        episode_lengths.append(steps)
    return episode_lengths

def q_learning_decentralized(env, num_episodes, alpha, gamma, epsilon):
    """ Múltiples agentes independientes, cada uno con su Q-table. """
    num_agents = env.num_of_agents
    q_tables = [defaultdict(lambda: {action: 1.0 for action in env.single_agent_action_space}) for _ in range(num_agents)]
    episode_lengths = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            steps += 1
            actions = []
            # Cada agente elige su acción de forma independiente
            for i in range(num_agents):
                if np.random.rand() < epsilon:
                    action = random.choice(env.single_agent_action_space)
                else:
                    action = max(q_tables[i][state], key=q_tables[i][state].get)
                actions.append(action)
            
            next_state, rewards, done = env.step(tuple(actions))
            
            # Cada agente actualiza su propia Q-table
            for i in range(num_agents):
                old_value = q_tables[i][state][actions[i]]
                next_max = max(q_tables[i][next_state].values())
                
                # El reward puede ser un escalar (cooperativo) o una tupla (competitivo)
                reward_i = rewards if np.isscalar(rewards) else rewards[i]

                new_value = old_value + alpha * (reward_i + gamma * next_max - old_value)
                q_tables[i][state][actions[i]] = new_value
            
            state = next_state
        episode_lengths.append(steps)
    return episode_lengths

def process_and_plot(results, title):
    plt.figure(figsize=(14, 8))
    
    for label, lengths_runs in results.items():
        # Promediar a través de los runs
        mean_lengths_over_runs = np.mean(lengths_runs, axis=0)
        
        # Agrupar y promediar cada 100 episodios
        report_interval = 100
        num_windows = len(mean_lengths_over_runs) // report_interval
        
        avg_lengths = [np.mean(mean_lengths_over_runs[i*report_interval:(i+1)*report_interval]) for i in range(num_windows)]
        episodes_axis = [i * report_interval for i in range(num_windows)]
        
        plt.plot(episodes_axis, avg_lengths, label=label)
        
    plt.xlabel('Episodios')
    plt.ylabel('Largo Promedio de Episodio (cada 100 eps)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('hunter_env_results.png')
    plt.show()

if __name__ == '__main__':
    
    # --- Valores para una prueba rápida ---
    num_runs = 30
    num_episodes = 50000

    # Hiperparámetros
    gamma = 0.95
    epsilon = 0.1
    alpha = 0.1
    
    all_results = {}
    
    # Experimento 1: Centralized Cooperative
    print("Ejecutando Centralized Cooperative (CentralizedHunterEnv)...")
    env_centralized = CentralizedHunterEnv()
    centralized_runs = []
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}", end='\r')
        lengths = q_learning_centralized(env_centralized, num_episodes, alpha, gamma, epsilon)
        centralized_runs.append(lengths)
    all_results['Centralized Cooperative'] = centralized_runs
    print("\n¡Completado!")

    # Experimento 2: Decentralized Cooperative
    print("\nEjecutando Decentralized Cooperative (HunterEnv)...")
    env_decentralized_coop = HunterEnv()
    decentralized_coop_runs = []
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}", end='\r')
        lengths = q_learning_decentralized(env_decentralized_coop, num_episodes, alpha, gamma, epsilon)
        decentralized_coop_runs.append(lengths)
    all_results['Decentralized Cooperative'] = decentralized_coop_runs
    print("\n¡Completado!")

    # Experimento 3: Decentralized Competitive
    print("\nEjecutando Decentralized Competitive (HunterAndPreyEnv)...")
    env_decentralized_comp = HunterAndPreyEnv()
    decentralized_comp_runs = []
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}", end='\r')
        lengths = q_learning_decentralized(env_decentralized_comp, num_episodes, alpha, gamma, epsilon)
        decentralized_comp_runs.append(lengths)
    all_results['Decentralized Competitive'] = decentralized_comp_runs
    print("\n¡Completado!")

    process_and_plot(all_results, 'Rendimiento en Dominios Multi-Agente')