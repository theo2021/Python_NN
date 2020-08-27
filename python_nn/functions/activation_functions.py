import numpy as np


class Sigmoid:
    def __init__(self, slope=1):
        if type(slope) != type(1):
            self.slope = 1
        else:
            self.slope = slope

    def func(self, x_vec):
        return 2.0 / (1.0 + np.exp(-self.slope*x_vec)) - 1

    def dev_func(self, x_vec):
        return ((1 + self.func(x_vec)) * (1 - self.func(x_vec)) / 2)*self.slope


class Softmax:

    @staticmethod
    def func(w_vec):
        return np.exp(w_vec)/np.outer(np.sum(np.exp(w_vec), axis=1), np.ones(w_vec.shape[1]))

    @staticmethod
    def dev_func(x_vec):
        ###normally
        # p_i = self.func(x_vec)
        # return np.diag(p_i) - np.outer(p_i, p_i)
        ### because its always used with crossentropy derivative included in crossentropy
        return np.ones(x_vec.shape)


class Relu:

    @staticmethod
    def func(x_vec):
        return (x_vec > 0)*1*x_vec

    @staticmethod
    def dev_func(x_vec):
        return (x_vec > 0)*1

class Tanh:

    @staticmethod
    def func(x_vec):
        return np.tanh(x_vec)

    @staticmethod
    def dev_func(x_vec):
        return 1 - np.tanh(x_vec)**2


class Linear:

    @staticmethod
    def func(x_vec):
        return x_vec

    @staticmethod
    def dev_func(x_vec):
        return np.ones_like(x_vec)


def activation_handler(func_string, **args):
    if func_string == 'linear':
        return Linear()
    elif func_string == "relu":
        return Relu()
    elif func_string == "sigmoid":
        return Sigmoid(**args)
    elif func_string == "tanh":
        return Tanh()
    elif func_string == "softmax":
        return Softmax()
