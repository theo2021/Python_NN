import numpy as np
from .layer import Layer
from ..functions.activation_functions import activation_handler


class ConvLayer1D(Layer):
    def __init__(self, previous_layer, input_dims, filter_len, filter_num, function='linear', stride=1, padding=0, initializer='XavieNormal'):
        input_dims = (input_dims[0], input_dims[1] + 2*padding)
        super().__init__('ConvLayer1D', {'filters': (filter_num, input_dims[0]*filter_len), 'bias': filter_num}, initializer)
        self.args = [input_dims, filter_len, filter_num, stride, padding]
        self.filter_out_size = ((input_dims[1] - filter_len) // stride + 1)
        self.create_padded()
        self.transform_bias()
        self.activation = activation_handler(function)
        self.params = len(self['filters'].reshape(-1)) + len(self['bias'])

    def create_padded(self):
        input_dims, filter_len, filter_num, stride, padding = self.args
        filter_out_size = self.filter_out_size
        self.filters_padded = np.zeros((filter_out_size*filter_num, input_dims[0]*input_dims[1]))
        for i in range(filter_out_size):
            pos_start = i*stride*input_dims[0]
            self.filters_padded[i*filter_num:i*filter_num+filter_num, pos_start: pos_start+self['filters'].shape[1]] = self['filters']

    def transform_bias(self):
        self.bias_trans = np.empty(0)
        for i in range(self.filter_out_size):
            self.bias_trans = np.hstack((self.bias_trans, self['bias']))

    def output_dims(self):
        return self['filters'].shape[0], self.filter_out_size

    def pad_input(self, inp):
        input_dims, filter_len, filter_num, stride, padding = self.args
        if padding == 0:
            return inp
        pad = np.zeros((inp.shape[0], input_dims[0]*padding))
        new_input = np.hstack((pad, inp, pad))
        return new_input

    def unpad_input(self, inp):
        input_dims, filter_len, filter_num, stride, padding = self.args
        if padding == 0:
            return inp
        return inp[:, input_dims[0] * padding: - input_dims[0] * padding]

    def edit_params(self, array):
        self.super().edit_params(array)
        self.create_padded()

    def feed(self, input):
        before_activation = self.pad_input(input) @ self.filters_padded.T + self.bias_trans
        return before_activation, self.activation.func(before_activation)

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
        self.update_params({'filters': (arr.T@ prob_activation.reshape(-1, filter_num)).T/n, 'bias': new_bias}, update_params)
        self.create_padded()
        self.transform_bias()
        return ds
