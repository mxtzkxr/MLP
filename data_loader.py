import math
import random

import numpy as np


class Loader:
    def __init__(self,
                 dimensions = 2,
                 train_percent = 85.0):
        self.__tp = train_percent
        self.__train_selection, self.__test_selection = self.__load_data(dimensions)

    def __load_data(self, dim):
        data = self.__get2DData() if dim == 2 else self.__get3DData()
        # количество примеров
        ln = len(data)
        ln_test_selection = int(ln * (1 - self.__tp / 100))
        ln_train_selection = ln - ln_test_selection

        random.shuffle(data)
        return sorted(data[:ln_train_selection]), sorted(data[ln_train_selection:])

    def __get2DData(self):
        return [
            [
                [i / 10],
                [math.cos(i / 10) + random.random() * 0.2 - 0.1]
            ]
            for i in range(-60, 61)
        ]

    def __get3DData(self):
        return [
            [
                [i / 10, i / 20],
                [(math.cos(i / 10) + math.sin(i/20)) + random.random() * 0.2 - 0.1]
            ]
            for i in range(-60, 61)
        ]

    def get_train_inp(self):
        return np.array([i[0] for i in self.__train_selection])

    def get_train_out(self):
        return np.array([i[1] for i in self.__train_selection])

    def get_test_inp(self):
        return np.array([i[0] for i in self.__test_selection])

    def get_test_out(self):
        return np.array([i[1] for i in self.__test_selection])