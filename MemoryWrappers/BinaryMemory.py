from Environments.AbstractEnv import AbstractEnv


class BinaryMemory(AbstractEnv):

    def __init__(self, env, num_of_bits):
        self.__env = env
        self.__num_of_bits = num_of_bits
        self.__env_state = None
        self.__memory_state = None

    @property
    def action_space(self):
        env_actions = self.__env.action_space
        memory_actions = list(range(2**self.__num_of_bits))
        return [(env_action, memory_action) for env_action in env_actions for memory_action in memory_actions]

    def reset(self):
        self.__env_state = self.__env.reset()
        self.__memory_state = 0
        return self.__get_state()

    def __get_state(self):
        return self.__env_state, self.__memory_state

    def step(self, action):
        env_action, mem_action = action
        self.__env_state, r, done = self.__env.step(env_action)
        self.__memory_state = mem_action
        return self.__get_state(), r, done

    def show(self):
        self.__env.show()
        print(f"Memory: {self.__memory_state}")
