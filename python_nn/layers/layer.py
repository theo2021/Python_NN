import numpy as np
from ..initializers import initializer_handler


class Layer(object):

    def __init__(self, name, variables, initializer='XavieNormal'):
        self.storage = {}
        self.storage['parameters'] = {}
        self.storage['derivatives'] = {}
        self.params = 0
        for par, shape in variables.items():
            self.storage['parameters'][par] = np.zeros(shape)
            self.storage['derivatives'][par] = np.zeros(shape)
            self.params += np.prod(shape)
        self.name = name
        self.initializer = initializer_handler[initializer]
        self.__initialize()
    
    def __initialize(self, initializer=None):
        if initializer is None:
            initializer = self.initializer
        for par in self.storage['parameters'].keys():
            self.storage['parameters'][par] = initializer(self.storage['parameters'][par].shape)
    
    def __getitem__(self, par):
        return self.storage['parameters'][par]

    def dev(self, par):
        return self.storage['derivatives'][par]

    def get_params_vec(self):
        buff = np.empty(0)
        for val in self.storage['parameters'].values():
            buff = np.hstack(buff, val.reshape(-1))
        return buff

    def get_dev_vec(self):
        buff = np.empty(0)
        for val in self.storage['derivatives'].values():
            buff = np.hstack(buff, val.reshape(-1))
        return buff

    def edit_params(self, array):
        pos = 0
        for par in self.storage['parameters'].keys():
            par_shape = self.storage['parameters'][par].shape
            self.storage['parameters'][par] = array[pos:np.prod(par_shape)].reshape(par_shape)
            pos += np.prod(par_shape)
    
    def update_params(self, devs, update_params):
        for par, val in devs.items():
            val_with_reg = val + update_params['reg'].dev(self[par])
            self.storage['parameters'][par] -= update_params['optimizer'](self.who_am_i, par, val_with_reg)
            self.storage['derivatives'][par] = val_with_reg