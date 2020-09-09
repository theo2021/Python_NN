from .layer import Layer
from ..functions.activation_functions import activation_handler
import numpy as np


class Dense(Layer):

    def __init__(self, previous_layer, input_dims, nodes, function='linear', bias_weight=1, initializer='XavieNormal'):
        super().__init__('Dense', {'W': (nodes, previous_layer), 'b': nodes}, initializer)
        self.recognizer = 'Dense'
        self.activation = activation_handler(function)
        self.bias_weight = bias_weight

    def output_dims(self):
        return None

    def feed(self, input):
        before_activation = (input @ self['W'].T) + self.bias_weight * self['b']
        return before_activation, self.activation.func(before_activation)

    def update_return(self, delta, before_activation, previous_layer, update_params):
        prob_activation = delta * self.activation.dev_func(before_activation)
        dw = prob_activation.T @ previous_layer/previous_layer.shape[0]
        db = np.mean(prob_activation.T, axis=1)
        ds = prob_activation @ self['W']
        self.update_params({'W': dw, 'b': db}, update_params)
        return ds
