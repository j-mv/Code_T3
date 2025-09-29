import numpy as np
import random
from collections import defaultdict
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv

# Implementación de Dyna-Q
def dyna_q(env, num_episodes, alpha, gamma, epsilon, planning_steps):
    q_table = defaultdict(lambda: {action: 0 for action in env.action_space})
    model = {}  # Modelo (s, a) -> (r, s')
    
    returns = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            if state not in q_table:
                q_table[state] = {action: 0 for action in env.action_space}

            if np.random.rand() < epsilon:
                action = random.choice(env.action_space)
            else:
                action = max(q_table[state], key=q_table[state].get)

            next_state, reward, done = env.step(action)
            episode_return += reward

            # Actualización de Q-Learning (aprendizaje directo)
            if next_state not in q_table:
                q_table[next_state] = {action: 0 for action in env.action_space}
            
            old_value = q_table[state][action]
            next_max = max(q_table[next_state].values())
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state][action] = new_value

            # Actualización del modelo
            if state not in model:
                model[state] = {}
            model[state][action] = (reward, next_state)
            
            # Planificación (planning)
            for _ in range(planning_steps):
                # Seleccionar un estado y acción previamente observados al azar
                s_rand = random.choice(list(model.keys()))
                a_rand = random.choice(list(model[s_rand].keys()))
                
                r_model, s_prime_model = model[s_rand][a_rand]
                
                # Actualizar Q-table con la experiencia imaginada
                old_value_plan = q_table[s_rand][a_rand]
                next_max_plan = max(q_table[s_prime_model].values())
                new_value_plan = old_value_plan + alpha * (r_model + gamma * next_max_plan - old_value_plan)
                q_table[s_rand][a_rand] = new_value_plan

            state = next_state
        returns.append(episode_return)
        
    return np.mean(returns)

# Implementación de R-Max
def r_max(env, num_episodes, gamma, m, r_max_val):
    # Inicialización
    q_table = defaultdict(lambda: {action: r_max_val for action in env.action_space})
    r_table = defaultdict(lambda: {action: 0 for action in env.action_space})
    t_table = defaultdict(lambda: {action: defaultdict(int) for action in env.action_space})
    n_table = defaultdict(lambda: {action: 0 for action in env.action_space})
    
    returns = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            action = max(q_table[state], key=q_table[state].get)
            
            next_state, reward, done = env.step(action)
            episode_return += reward
            
            # Actualizar el modelo si es necesario
            if n_table[state][action] < m:
                n_table[state][action] += 1
                r_table[state][action] += (reward - r_table[state][action]) / n_table[state][action]
                t_table[state][action][next_state] += 1
                
                if n_table[state][action] == m:
                    # El estado-acción es conocido, actualizar Q-values mediante value iteration
                    for _ in range(100): # Iteraciones de planificación
                        for s_plan in n_table:
                            for a_plan in n_table[s_plan]:
                                if n_table[s_plan][a_plan] >= m:
                                    # Calcular la transición más probable
                                    s_prime_plan = max(t_table[s_plan][a_plan], key=t_table[s_plan][a_plan].get)
                                    
                                    q_val = r_table[s_plan][a_plan] + gamma * max(q_table[s_prime_plan].values())
                                    q_table[s_plan][a_plan] = q_val
            
            state = next_state
        returns.append(episode_return)

    return np.mean(returns)

def run_experiments():
    env = EscapeRoomEnv()
    num_runs = 5
    num_episodes = 20
    gamma = 1.0
    
    # Parámetros Dyna-Q
    alpha_dyna = 0.5
    epsilon_dyna = 0.1
    planning_steps_list = [0, 1, 10, 100, 1000, 10000]
    
    # Parámetros R-Max
    m_rmax = 5 # Umbral para considerar un estado-acción como "conocido"
    r_max_val = 1 # Valor optimista para la recompensa
    
    results = defaultdict(list)
    
    for planning_steps in planning_steps_list:
        run_returns = []
        for _ in range(num_runs):
            ret = dyna_q(env, num_episodes, alpha_dyna, gamma, epsilon_dyna, planning_steps)
            run_returns.append(ret)
        results['Dyna'].append(np.mean(run_returns))

    run_returns_rmax = []
    for _ in range(num_runs):
        ret = r_max(env, num_episodes, gamma, m_rmax, r_max_val)
        run_returns_rmax.append(ret)
    results['RMax'] = np.mean(run_returns_rmax)

    print("Resultados:")
    print("-----------------------------------------")
    print("Algoritmo\t| Planning Steps\t| Retorno Medio")
    print("-----------------------------------------")
    for i, ps in enumerate(planning_steps_list):
        print(f"Dyna-Q\t\t| {ps}\t\t\t| {results['Dyna'][i]:.2f}")
    print(f"R-Max\t\t| N/A\t\t\t| {results['RMax']:.2f}")
    print("-----------------------------------------")

if __name__ == '__main__':
    print("Ejecutando experimentos con Dyna-Q y R-Max en EscapeRoomEnv...")
    run_experiments()