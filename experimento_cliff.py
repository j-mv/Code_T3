import numpy as np
import matplotlib.pyplot as plt
from Environments.SimpleEnvs.CliffEnv import CliffEnv

def q_learning(env, num_episodes, alpha, gamma, epsilon):
    q_table = {}
    returns = []

    for _ in range(num_episodes):
        state = env.reset()
        if state not in q_table:
            q_table[state] = {action: 0 for action in env.action_space}
        
        episode_return = 0
        done = False
        
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space)
            else:
                action = max(q_table[state], key=q_table[state].get)

            next_state, reward, done = env.step(action)
            if next_state not in q_table:
                q_table[next_state] = {action: 0 for action in env.action_space}

            old_value = q_table[state][action]
            next_max = max(q_table[next_state].values())
            
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state][action] = new_value
            
            state = next_state
            episode_return += reward
            
        returns.append(episode_return)
        
    return returns

def sarsa(env, num_episodes, alpha, gamma, epsilon, n=1):
    q_table = {}
    returns = []

    for _ in range(num_episodes):
        state = env.reset()
        if state not in q_table:
            q_table[state] = {action: 0 for action in env.action_space}
        
        if np.random.rand() < epsilon:
            action = np.random.choice(env.action_space)
        else:
            action = max(q_table[state], key=q_table[state].get)
        
        episode_return = 0
        done = False
        
        states = [state]
        actions = [action]
        rewards = [0]
        
        T = float('inf')
        t = 0
        
        while True:
            if t < T:
                next_state, reward, done = env.step(actions[t])
                
                states.append(next_state)
                rewards.append(reward)
                episode_return += reward
                
                if done:
                    T = t + 1
                else:
                    if next_state not in q_table:
                        q_table[next_state] = {action: 0 for action in env.action_space}
                    
                    if np.random.rand() < epsilon:
                        next_action = np.random.choice(env.action_space)
                    else:
                        next_action = max(q_table[next_state], key=q_table[next_state].get)
                    actions.append(next_action)

            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (gamma**(i - tau - 1)) * rewards[i]
                if tau + n < T:
                    G += (gamma**n) * q_table[states[tau + n]][actions[tau + n]]
                
                q_table[states[tau]][actions[tau]] += alpha * (G - q_table[states[tau]][actions[tau]])

            if tau == T - 1:
                break
            
            t += 1
            
        returns.append(episode_return)
        
    return returns

def plot_results(returns_q, returns_sarsa, returns_4_step_sarsa):
    plt.figure(figsize=(12, 8))
    
    plt.plot(np.mean(returns_q, axis=0), label='Q-Learning')
    plt.plot(np.mean(returns_sarsa, axis=0), label='Sarsa')
    plt.plot(np.mean(returns_4_step_sarsa, axis=0), label='4-Step Sarsa')
    
    plt.ylim(-200, 0)
    plt.xlabel('Episodios')
    plt.ylabel('Retorno Promedio')
    plt.title('Rendimiento en CliffEnv')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    env = CliffEnv()
    num_episodes = 500
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1
    num_experiments = 100

    all_returns_q = []
    all_returns_sarsa = []
    all_returns_4_step_sarsa = []

    for _ in range(num_experiments):
        returns_q = q_learning(env, num_episodes, alpha, gamma, epsilon)
        returns_sarsa = sarsa(env, num_episodes, alpha, gamma, epsilon)
        returns_4_step_sarsa = sarsa(env, num_episodes, alpha, gamma, epsilon, n=4)
        
        all_returns_q.append(returns_q)
        all_returns_sarsa.append(returns_sarsa)
        all_returns_4_step_sarsa.append(returns_4_step_sarsa)

    plot_results(all_returns_q, all_returns_sarsa, all_returns_4_step_sarsa)