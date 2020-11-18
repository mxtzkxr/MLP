from data_loader import Loader
import numpy as np

class MLP:
    __eta = 0.005

    def __init__(self, ld: Loader, neurons: tuple = (3, 2)):
        self.__layers = 2 + len(neurons)
        inp = ld.get_train_inp()
        out = ld.get_train_out()
        nN = [len(inp[0]), len(out[0])]
        #инфа о количестве нейронов
        self.__neurons = np.insert(nN, 1, neurons)
        self.__inp = np.array(inp)
        self.__out = np.array(out)
        self.__tst_inp = np.array(ld.get_test_inp())
        self.__tst_out = np.array(ld.get_test_out())
        self.__weights = [np.random.rand(
            self.__neurons[i + 1] + (0 if i == self.__layers - 2 else 1),
            self.__neurons[i] + 1
        ) for i in range(self.__layers - 1)]

    def s(self, x):
            return np.array((np.tanh(x)))

    def nonlinear(self, x, der=False):
            if der:
                return np.array(1 - self.s(x) ** 2)
            return self.s(x)

    def linear(self, x, der=False):
            if der:
                return np.array(1)
            return np.array(x)

    def learn(self):
        epsilon = 0.002
        err_n = epsilon + 1
        train_err = epsilon + 1
        k = 0
        epochs = 10000

        inp = self.__inp
        out = self.__out

        v = np.array([None for i in range(self.__layers - 1)])
        l_out = np.array([None for i in range(self.__layers)])
        deltas = np.array([None for i in range(self.__layers - 1)])

        while k < epochs and err_n > epsilon:
            err_n = 0
            train_err = 0
            k += 1
            for i in range(len(inp)):
                l_out[0] = np.array([np.insert(inp[i], 0, 1)])

                for j in range(self.__layers-2):
                    v[j] = l_out[j].dot(self.__weights[j].T)
                    l_out[j+1] = self.nonlinear(v[j])
                v[self.__layers-2] = l_out[self.__layers-2].dot(self.__weights[self.__layers-2].T)
                l_out[self.__layers-1] = self.linear(v[self.__layers-2])
                error = (out[i] - l_out[self.__layers-1])
                train_err += 0.5*error**2
                deltas[self.__layers-2] = np.array([error[0]*self.linear(v[self.__layers-2], True)])
                for j in range(self.__layers-2, 0, -1):
                    deltas[j-1] = deltas[j].dot(self.__weights[j]) * self.nonlinear(v[j-1], True)
                dW = [self.__eta * l_out[j].T.dot(deltas[j]).T for j in range(self.__layers-1)]
                for j in range(self.__layers-1):
                    self.__weights[j] += dW[j]
                if k % 1000 == 0:
                    print(l_out[0])
                    print(':')
                    print(l_out[self.__layers-1])
            train_err /= len(inp)
            outt = self.calc(self.__tst_inp)
            ln = len(outt)
            soutt = np.array([self.__tst_out[i][0] for i in range(len(self.__tst_out))])
            err_n = np.sum(0.5 * (soutt - outt) ** 2) / ln

    def calc(self, inps):
        outs = np.array([])
        # Для каждого входного значения
        for i in range(len(inps)):
            inp = np.array([np.insert(inps[i], 0, 1)])
            # Прямой проход по сети (все слои, кроме последнего)
            for lr in range(self.__layers - 2):
                inp = self.nonlinear(np.dot(inp, self.__weights[lr].T))
            # Получение результата на последнем слое
            # и добавлени его в массив выходов
            outs = np.append(outs,
                             self.linear(np.dot(inp, self.__weights[self.__layers - 2].T))
                             )
        return outs