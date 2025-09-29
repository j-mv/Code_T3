from collections import deque

from Environments.AbstractEnv import AbstractEnv


class KOrderMemory(AbstractEnv):

    def __init__(self, env, memory_size):
        self.__env = env
        self.__memory_size = memory_size
        self.__memory = None

    @property
    def action_space(self):
        return self.__env.action_space

    def reset(self):
        s = self.__env.reset()
        self.__memory = deque(maxlen=self.__memory_size)
        self.__memory.append(s)
        return tuple(self.__memory)

    def step(self, action):
        s, r, done = self.__env.step(action)
        self.__memory.append(s)
        return tuple(self.__memory), r, done

    def show(self):
        self.__env.show()
        print(f"Memory: {self.__memory}")
