from .layer import Layer
from ..functions.activation_functions import activation_handler
import numpy as np



# class Dense:

#     def __init__(self, previous_layer, input_dims, nodes, function='linear', bias_weight=1):
#         self.recognizer = 'Dense'
#         self.W = np.random.randn(nodes, previous_layer) / previous_layer
#         self.bias = np.random.rand(nodes) / previous_layer
#         self.activation = activation_handler(function)
#         self.bias_weight = bias_weight
#         self.params = len(self.W.reshape(-1)) + len(self.bias.reshape(-1))
#         self.dw, self.db = 0, 0

#     def output_dims(self):
#         return None

#     def edit_params(self, array):
#         lw = len(self.W.reshape(-1))
#         self.W = array[:lw].reshape(self.W.shape)
#         self.bias = array[lw:].reshape(self.bias.shape)

#     def get_params_vec(self):
#         return np.hstack((self.W.reshape(-1), self.bias.reshape(-1)))

#     def get_dev_vec(self):
#         return np.hstack((self.dw.reshape(-1), self.db.reshape(-1)))

#     def feed(self, input):
#         before_activation = (input @ self.W.T) + self.bias_weight * self.bias
#         return before_activation, self.activation.func(before_activation)

#     def ds(self, delta, before_activation):
#         return delta * self.activation.dev_func(before_activation) @ self.W

#     def update_return(self, delta, before_activation, previous_layer, update_params):
#         prob_activation = delta * self.activation.dev_func(before_activation)
#         dw = prob_activation.T @ previous_layer/previous_layer.shape[0]
#         db = np.mean(prob_activation.T, axis=1)
#         ds = prob_activation @ self.W
#         self.update_params(dw, db, update_params)
#         return ds

#     def update_params(self, dw, db, update_params):
#         tmp = dw + update_params['reg'].dev(self.W)
#         self.W -= update_params['optimizer'](self.who_am_i, 'W', tmp)
#         self.bias -= update_params['optimizer'](self.who_am_i, 'bias', db)
#         self.db = db
#         self.dw = tmp

#     def reg_error(self, update_params):
#         return update_params['reg'].cost(self.W)

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
