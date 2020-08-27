import numpy as np


class MeanSquaredError:

    @staticmethod
    def func(labels, outputs):
        return np.mean(np.power(labels-outputs, 2), axis=1)/2

    @staticmethod
    def dev_func(labels, outputs):
        return outputs - labels


class Entropy:

    def __init__(self, class_weights=None):
        self.weights = class_weights

    def func(self, labels, outputs):
        selection = np.argmax(labels, axis=1)
        sum_ = 0
        weights = np.ones(labels.shape[1])
        if self.weights is not None:
            weights = self.weights
        for sel, out, weight in zip(selection, outputs, weights[selection]):
            sum_ += - weight * np.log(out[sel]) / outputs.shape[0]
        return sum_

    def dev_func(self, labels, outputs):
        # selection = np.argmax(labels, axis=1)
        # tmp = np.zeros(outputs.shape)
        # for sel, out in zip(selection, outputs):
        #     tmp[sel] = -1/out[sel]
        # always with softmax
        weights = np.ones(labels.shape[1])
        if self.weights is not None:
            weights = self.weights
        weight_matrix = weights[np.argmax(labels, axis=1)]
        return weight_matrix.reshape(-1, 1) * (outputs - labels)


# class SVM:
#
#     def __call__(self, labels, outputs):
#         val = labels.dot(outputs)
#         tmp = (outputs - val + 1)*(1 - labels)
#         return np.sum((tmp > 0)*1*tmp)
#
#     def dev(self, labels, outputs):
#         val = labels.dot(outputs)
#         tmp = (outputs - val + 1) * (1 - labels)
#         tmp = (tmp > 0)*1
#         return tmp - labels*np.sum(tmp)


def los_handler(func_string, *args):
    if func_string == 'mse':
        return MeanSquaredError()
    elif func_string == "entropy":
        return Entropy(*args)
    # elif func_string == "svm":
    #     return SVM()

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
        return self.lamb*2*w
        return np.hstack([self.lamb*2*w, np.zeros((w.shape[0], 1))])


def reg_handler(reg_str=-1, **args):
    if reg_str == 'ridge':
        return RidgeRegularizer(**args)
    else:
        return NoRegularizer()


