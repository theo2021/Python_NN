import numpy as np


class MeanSquaredError:

    def __call__(self, labels, outputs):
        return np.mean(np.power(labels-outputs, 2))

    def dev(self, labels, outputs):
        return outputs - labels


class Entropy:

    def __call__(self, labels, outputs):
        selection = np.argmax(labels)
        return -np.log(outputs[selection])

    def dev(self, labels, outputs):
        selection = np.argmax(labels)
        tmp = np.zeros(outputs.shape)
        tmp[selection] = -1/outputs[selection]
        return tmp


def los_handler(func_string, *args):
    if func_string == 'mse':
        return MeanSquaredError()
    elif func_string == "entropy":
        return Entropy()

# Regularization


class NoRegularizer:

    def cost(self, w):
        return 0

    def dev(self, w):
        return 0


class RidgeRegularizer:
    def __init__(self, lamb=0.5):
        self.lamb = lamb

    def cost(self, w):
        return self.lamb*np.sum(w**2)

    def dev(self, w):
        return np.hstack([self.lamb*2*w, np.zeros((w.shape[0], 1))])


def reg_handler(reg_str=-1, **args):
    if reg_str == 'ridge':
        return RidgeRegularizer(**args)
    else:
        return NoRegularizer()
