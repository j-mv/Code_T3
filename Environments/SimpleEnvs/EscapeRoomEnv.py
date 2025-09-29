from Environments.GridEnv import GridEnv


class EscapeRoomEnv(GridEnv):

    def __init__(self):
        super().__init__(height=10, width=25)
        self.__key = (0, 0)
        self.__door = (0, self._width - 1)
        self.__nails = [(i, j) for i in range(self._height // 2) for j in range(3, self._width - 3)]
        self.__has_key = None
        self.__agents_location = None

    def reset(self):
        self.__has_key = False
        self.__agents_location = (self._height - 1, self._width - 1)
        return self.__agents_location + (self.__has_key,)

    def step(self, action):
        self.__agents_location = super()._get_new_location(self.__agents_location, action)
        self.__has_key = self.__has_key or self.__agents_location == self.__key
        state = self.__agents_location + (self.__has_key,)
        at_nail = self.__agents_location in self.__nails
        at_door = self.__agents_location == self.__door
        done = at_door and self.__has_key
        reward = -10 if at_nail else -1
        return state, reward, done

    def _get_location_letter(self, location):
        if location == self.__agents_location:
            return "A"
        if location == self.__key and not self.__has_key:
            return "K"
        if location == self.__door:
            return "D"
        if location in self.__nails:
            return "N"
        return " "
