from Environments.PartiallyObservableEnvs.InvisibleDoorEnv import InvisibleDoorEnv
from MainSimpleEnvs import show, get_action_from_user, play_simple_env
from MemoryWrappers.BinaryMemory import BinaryMemory
from MemoryWrappers.KOrderMemory import KOrderMemory


def play_env_with_binary_memory():
    num_of_bits = 1
    env = InvisibleDoorEnv()
    env = BinaryMemory(env, num_of_bits)

    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down"}
    key2memory = {str(i): i for i in range(2**num_of_bits)}
    s = env.reset()
    show(env, s)
    done = False
    while not done:
        print("Environment action: ", end="")
        env_action = get_action_from_user(key2action)
        print(f"Memory action ({', '.join(key2memory.keys())}): ", end="")
        mem_action = get_action_from_user(key2memory)
        action = env_action, mem_action
        s, r, done = env.step(action)
        show(env, s, r)


def play_env_with_k_order_memory():
    memory_size = 2
    env = InvisibleDoorEnv()
    env = KOrderMemory(env, memory_size)
    play_simple_env(env)


def play_env_without_extra_memory():
    env = InvisibleDoorEnv()
    play_simple_env(env)


if __name__ == '__main__':
    play_env_without_extra_memory()
    play_env_with_k_order_memory()
    play_env_with_binary_memory()


