from __future__ import absolute_import
import numpy as np

def limit(data, intensity=3):
    return np.clip(data, -intensity, intensity)

class SGD:
    def __init__(self, layers, learning_rate, momentum):
        self.lr = learning_rate
        self.momentum = momentum
        self.memory = {}
        for i in range(layers):
            self.memory[i] = {}

    def __call__(self, layer, uid, parameter):
        if uid not in self.memory[layer].keys():
            self.memory[layer][uid] = 0
        out = self.lr * (self.memory[layer][uid]*self.momentum + parameter)
        self.memory[layer][uid] = parameter
        return out

class AdaGrad:
    def __init__(self, layers, learning_rate, gamma=0.9):
        self.lr = learning_rate
        self.gamma = gamma
        self.memory = {}
        for i in range(layers):
            self.memory[i] = {}

    def __call__(self, layer, uid, parameter):
        parameter = limit(parameter)
        if uid not in self.memory[layer].keys():
            self.memory[layer][uid] = 0
        self.memory[layer][uid] = self.gamma*self.memory[layer][uid] + (1 - self.gamma)*(parameter**2)
        out = self.lr * (parameter)/(np.sqrt(self.memory[layer][uid] + 10**(-8)))
        return out

def opt_handler(opt_str=-1, **args):
    if opt_str == 'SGD':
        return SGD(**args)
    elif opt_str == 'AdaGrad':
        return AdaGrad(**args)
    else:
        print("Wrong optimizer value!")
        return None

