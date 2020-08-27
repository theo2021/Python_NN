import numpy as np
from python_nn.activation_functions import *
from scipy import sparse


class Dense:

    def __init__(self, previous_layer, input_dims, nodes, function='linear', bias_weight=1):
        self.recognizer = 'Dense'
        self.W = np.random.randn(nodes, previous_layer) / previous_layer
        self.bias = np.random.rand(nodes) / previous_layer
        self.activation = activation_handler(function)
        self.bias_weight = bias_weight
        self.params = len(self.W.reshape(-1)) + len(self.bias.reshape(-1))
        self.dw, self.db = 0, 0

    def output_dims(self):
        return None

    def edit_params(self, array):
        lw = len(self.W.reshape(-1))
        self.W = array[:lw].reshape(self.W.shape)
        self.bias = array[lw:].reshape(self.bias.shape)

    def get_params_vec(self):
        return np.hstack((self.W.reshape(-1), self.bias.reshape(-1)))

    def get_dev_vec(self):
        return np.hstack((self.dw.reshape(-1), self.db.reshape(-1)))

    def feed(self, input):
        before_activation = (input @ self.W.T) + self.bias_weight * self.bias
        return before_activation, self.activation.func(before_activation)

    def ds(self, delta, before_activation):
        return delta * self.activation.dev_func(before_activation) @ self.W

    def update_return(self, delta, before_activation, previous_layer, update_params):
        prob_activation = delta * self.activation.dev_func(before_activation)
        dw = prob_activation.T @ previous_layer/previous_layer.shape[0]
        db = np.mean(prob_activation.T, axis=1)
        ds = prob_activation @ self.W
        self.update_params(dw, db, update_params)
        return ds

    def update_params(self, dw, db, update_params):
        tmp = dw + update_params['reg'].dev(self.W)
        self.W -= update_params['optimizer'](self.who_am_i, 'W', tmp)
        self.bias -= update_params['optimizer'](self.who_am_i, 'bias', db)
        self.db = db
        self.dw = tmp

    def reg_error(self, update_params):
        return update_params['reg'].cost(self.W)


class ConvLayer1D:
    def __init__(self, previous_layer, input_dims, filter_len, filter_num, function='linear', stride=1, padding=0):
        self.recognizer = 'ConvLayer1D'
        input_dims = (input_dims[0], input_dims[1] + 2*padding)
        self.args = [input_dims, filter_len, filter_num, stride, padding]
        self.filters = np.random.randn(filter_num, input_dims[0]*filter_len)/np.sqrt((input_dims[0]*filter_len))*2
        self.bias = np.random.randn(filter_num) / np.sqrt((input_dims[0]*filter_len))*2
        self.filter_out_size = ((input_dims[1] - filter_len) // stride + 1)
        self.create_padded()
        self.transform_bias()
        self.activation = activation_handler(function)
        self.dw = 0
        self.db = 0
        self.store = []
        self.params = len(self.filters.reshape(-1)) + len(self.bias)

    def output_dims(self):
        return self.filters.shape[0], self.filter_out_size

    def create_padded(self):
        input_dims, filter_len, filter_num, stride, padding = self.args
        filter_out_size = self.filter_out_size
        self.filters_padded = np.zeros((filter_out_size*filter_num, input_dims[0]*input_dims[1]))
        for i in range(filter_out_size):
            pos_start = i*stride*input_dims[0]
            self.filters_padded[i*filter_num:i*filter_num+filter_num, pos_start: pos_start+self.filters.shape[1]] = self.filters

    def transform_bias(self):
        self.bias_trans = np.empty(0)
        for i in range(self.filter_out_size):
            self.bias_trans = np.hstack((self.bias_trans, self.bias))

    def pad_input(self, input):
        input_dims, filter_len, filter_num, stride, padding = self.args
        if padding == 0:
            return input
        pad = np.zeros((input.shape[0], input_dims[0]*padding))
        new_input = np.hstack((pad, input, pad))
        return new_input

    def unpad_input(self, input):
        input_dims, filter_len, filter_num, stride, padding = self.args
        if padding == 0:
            return input
        return input[:, input_dims[0] * padding: - input_dims[0] * padding]

    def edit_params(self, array):
        self.filters = array[:-self.bias.shape[0]].reshape(self.filters.shape)
        self.bias = array[-self.bias.shape[0]:]
        self.create_padded()
        self.transform_bias()

    def get_params_vec(self):
        return np.hstack((self.filters.reshape(-1), self.bias))

    def get_dev_vec(self):
        return np.hstack((self.dw.reshape(-1), self.db))

    def feed(self, input):
        before_activation = self.pad_input(input) @ self.filters_padded.T + self.bias_trans
        return before_activation, self.activation.func(before_activation)

    def reg_error(self, update_params):
        return update_params['reg'].cost(self.filters)

    def update_return(self, delta, before_activation, previous_layer, update_params):
        previous_layer = self.pad_input(previous_layer)
        input_dims, filter_len, filter_num, stride, padding = self.args
        prob_activation = delta * self.activation.dev_func(before_activation)
        filter_size = filter_len * input_dims[0]
        filter_out_size = ((input_dims[1] - filter_len) // stride + 1)
        n = previous_layer.shape[0]
        arr = np.zeros((filter_out_size * n, filter_size))
        for i_s, sample in enumerate(previous_layer):
            pos = i_s * filter_out_size
            for i, j in enumerate(range(pos, pos + filter_out_size)):
                arr[j, :] = previous_layer[i_s, i*input_dims[0]*stride: i*input_dims[0]*stride + filter_size]
        ds = self.unpad_input(prob_activation @ self.filters_padded)
        new_bias = np.sum(prob_activation.reshape(-1, filter_num), axis=0)/n
        self.update_params((arr.T@ prob_activation.reshape(-1, filter_num)).T/n, new_bias, update_params)
        return ds


    def update_params(self, dw, db, update_params):
        tmp = dw + update_params['reg'].dev(self.filters)
        self.filters -= update_params['step']*(update_params['momentum']*self.dw + tmp)
        self.bias -= update_params['step']*(update_params['momentum']*self.db + db)
        self.db = db
        self.dw = tmp
        self.create_padded()
        self.transform_bias()

class Recurent:
    def __init__(self, previous_layer, input_dims, function='linear'):
        self.recognizer = 'Recurent'
        self.W = np.random.randn(previous_layer, previous_layer)/previous_layer
        self.activation = activation_handler(function)
        self.dw = 0
        self.h0 = np.zeros(previous_layer)
        self.h1 = np.zeros(previous_layer)
        self.remember = False
        self.params = previous_layer**2

    def output_dims(self):
        return None

    def h_init(self):
        self.h0 = np.zeros(self.W.shape[0])
        self.h1 = np.zeros(self.W.shape[0])

    def get_params_vec(self):
        return self.W.reshape(-1)

    def edit_params(self, array):
        self.W = array.reshape(self.W.shape)

    def get_dev_vec(self):
        return self.dw.reshape(-1)

    def feed(self, input):
        start_inp = self.h1
        before_activation = np.zeros_like(input)
        after_activation = np.zeros_like(input)
        for i, inp in enumerate(input):
            before_activation[i] = inp + self.W @ start_inp
            after_activation[i] = self.activation.func(before_activation[i])
            start_inp = after_activation[i]
        if self.remember:
            self.h0 = self.h1
            self.h1 = start_inp
        return before_activation, after_activation


    def update_return(self, delta, before_activation, previous_layer, update_params):
        prob_activation = self.activation.dev_func(before_activation)
        h = self.activation.func(before_activation)
        dh = np.zeros_like(delta)
        da = np.zeros_like(delta)
        dh[-1] = delta[-1]
        da[-1] = dh[-1]*prob_activation[-1]
        for i in range(len(delta)-2, -1, -1):
            dh[i] = da[i+1] @ self.W + delta[i]
            da[i] = dh[i] * prob_activation[i]

        h[1:] = h[:-1]
        h[0] = self.h0
        dw = (da.T @ h) / delta.shape[0]
        self.update_params(dw, update_params)
        return da

    def update_params(self, dw, update_params):
        tmp = dw + update_params['reg'].dev(self.W)
        self.W -= update_params['optimizer'](self.who_am_i, 'W', tmp)
        self.dw = tmp

    def reg_error(self, update_params):
        return update_params['reg'].cost(self.W)
