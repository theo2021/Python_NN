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

    def func(self, x_vec):
        if np.sum(np.exp(x_vec)) == 0:
            print(x_vec)
        return np.exp(x_vec)/np.sum(np.exp(x_vec))

    def dev_func(self, x_vec):
        p_i = self.func(x_vec)
        return np.diag(p_i) - np.outer(p_i, p_i)


class Relu:
    def func(self, x_vec):
        return (x_vec > 0)*1*x_vec

    def dev_func(self, x_vec):
        return (x_vec > 0)*1


class Linear:
    def func(self, x_vec):
        return x_vec

    def dev_func(self, x_vec):
        return 1


def activation_handler(func_string, **args):
    if func_string == 'linear':
        return Linear()
    elif func_string == "relu":
        return Relu()
    elif func_string == "sigmoid":
        return Sigmoid(**args)
    elif func_string == "softmax":
        return Softmax()
