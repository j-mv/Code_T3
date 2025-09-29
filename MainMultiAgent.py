from Environments.MultiAgentEnvs.HunterAndPreyEnv import HunterAndPreyEnv
from MainSimpleEnvs import show, get_action_from_user


def play_hunter_env():
    hunter_env = HunterAndPreyEnv()

    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down", '': "None"}
    num_of_agents = hunter_env.num_of_agents
    s = hunter_env.reset()
    show(hunter_env, s)
    done = False
    while not done:
        print("Hunter A: ", end="")
        hunter1 = get_action_from_user(key2action)
        print("Hunter B: ", end="")
        hunter2 = get_action_from_user(key2action)
        action = hunter1, hunter2
        if num_of_agents == 3:
            print("Prey: ", end="")
            prey = get_action_from_user(key2action)
            action = hunter1, hunter2, prey
        s, r, done = hunter_env.step(action)
        show(hunter_env, s, r)


if __name__ == '__main__':
    play_hunter_env()
