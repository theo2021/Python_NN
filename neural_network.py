import numpy as np
from scipy.stats import norm
from Libs.transformations import *
from Libs.activation_functions import *
from Libs.loss_functions import *


class Layer:
    def __init__(self, previous_layer, nodes, function='sigmoid', bias_weight=1, fixed_weights=False):
        self.W = np.random.rand(nodes, previous_layer) / previous_layer
        self.bias = np.random.rand(nodes) / previous_layer
        self.fixed_weights = fixed_weights
        if fixed_weights:
            self.W = np.ones(nodes, previous_layer)
            self.bias = np.zeros(nodes)
        self.function = activation_handler(function)
        self.last_signal = -1
        self.func = self.function.func
        self.dev = self.function.dev_func
        self.bias_weight = bias_weight
        self.dw = np.zeros((nodes, previous_layer+1))

    def f_b(self, x_vec):  # return bias too
        return np.append(self.func(x_vec), [1])

    def get_wb(self):
        return np.hstack((self.W, self.bias.reshape(-1, 1)))

    def update_weights(self, dw, momentum):
        if self.fixed_weights:
            return 0
        self.W += self.dw[:, :-1]*momentum + dw[:, :-1]  # the lab 1 says (1-momentum)* but i believe is incorrect
        self.bias += self.dw[:, -1]*momentum + dw[:, -1]
        self.dw = dw

    def feed(self, inp):
        self.last_signal = (self.W @ inp) + self.bias_weight*self.bias
        return self.last_signal, self.func(self.last_signal)


class NN:
    def __init__(self, input_num):
        # initialization of NN
        self.signal_dim = [input_num]
        self.layers = []
        self.transformations = []
        self.dw = []
        self.loss_function = los_handler('mse')
        self.regularizer = reg_handler()

    def add_transformation(self, trans):
        self.transformations.append(trans)
        o = trans(np.zeros(self.signal_dim[0]))
        self.signal_dim[0] = len(o)

    def add_layer(self, nodes, function='sigmoid', **args):
        self.layers.append(Layer(self.signal_dim[-1], nodes, function, **args))
        self.signal_dim.append(nodes)

    def compile(self, loss_function_str='mse', regularizer={'name': 'none', 'args': {}}):
        self.loss_function = los_handler(loss_function_str)
        self.regularizer = reg_handler(regularizer['name'], **regularizer['args'])

    def feed_forward(self, inp):
        tmp = inp
        for trans in self.transformations:
            tmp = trans(tmp)
        signal_list = [tmp]
        for layer in self.layers:
            signal, tmp = layer.feed(tmp)
            signal_list.append(signal)
        return signal_list, tmp

    def back_prob(self, prediction, target, signal_list):
        dw = []
        dev = self.layers[-1].dev(signal_list[-1])
        # if dev.ndim == 2:
        #     dev = target @ dev
        delta = self.loss_function.dev(target, prediction) @ dev
        for i in range(len(self.layers)-1, 0, -1):  # signal has length layers+1 (the input)
            dw.append(-np.outer(delta, self.layers[i-1].f_b(signal_list[i]))) #- self.regularizer.dev(self.layers[i].W))  # signal from previous layer
            delta = (self.layers[i].W.T @ delta)*self.layers[i-1].dev(signal_list[i])
        dw.append(-np.outer(delta, np.append(signal_list[0], [1]))) #- self.regularizer.dev(self.layers[0].W))
        dw.reverse()
        return dw

    def reg_error(self):
        er = 0
        for layer in self.layers:
            er += self.regularizer.cost(layer.W)
        return er

    def dw_batch(self, inputs, targets):
        #dw = [np.zeros((layer.W.shape[0], layer.W.shape[1] + 1)) for layer in self.layers]
        dw = [-self.regularizer.dev(layer.W) for layer in self.layers]
        for inp, trg in zip(inputs, targets):
            signals, res = self.feed_forward(inp)
            for i, dw_layer in enumerate(self.back_prob(res, trg, signals)):
                dw[i] += dw_layer/len(inputs)
        return dw

    def update_dw(self, new_dw, step, momentum):
        for i, new_dw_layer in enumerate(new_dw):
            self.layers[i].update_weights(step*new_dw_layer, momentum)

    def train_batch(self, inputs_start, targets_start, batch_size, epochs, step, momentum, learning_curve=False, test_set=None, test_targets=None):
        training_error = []
        test_error = []
        loops = int(len(inputs_start)/batch_size)
        for epoch in range(0, epochs):
            prev = 0
            tmp = np.random.permutation(inputs_start.shape[0])
            inputs = inputs_start[tmp]
            targets = targets_start[tmp]
            if learning_curve:
                error = 0#self.reg_error()
                for s, t in zip(inputs, targets):
                    error += self.loss_function(t, self.feed_forward(s)[1])
                error = error/len(inputs)
                training_error.append(error)
            error = 0#self.reg_error()
            if test_set is not None:
                for s, t in zip(test_set, test_targets):
                    error += self.loss_function(t, self.feed_forward(s)[1])
                error = error / len(test_set)
                test_error.append(error)
            for i in range(0, loops):
                inp_batch = inputs[prev: batch_size*(i+1)]
                trg_batch = targets[prev: batch_size*(i+1)]
                new_dw = self.dw_batch(inp_batch, trg_batch)
                self.update_dw(new_dw, step, momentum)
                prev = batch_size*(i+1)
            inp_batch = inputs[batch_size * loops:]
            trg_batch = targets[batch_size * loops:]
            new_dw = self.dw_batch(inp_batch, trg_batch)
            self.update_dw(new_dw, step, momentum)
        return np.array(training_error), np.array(test_error)

