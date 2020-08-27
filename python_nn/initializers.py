import numpy as np

class Initializer(object):

    def __init__(self, name):
        self.name = name

    def __call__(self, dims):
        raise NotImplementedError

class HeNormal(Initializer):

    def __init__(self):
        super().__init__('HeNormal')
    
    def __call__(self, dims):
        return np.random.randn(*dims)*np.sqrt(2/dims[0])
    
class XavieNormal(Initializer):

    def __init__(self):
        super().__init__('XavieNormal')
    
    def __call__(self, dims):
        return np.random.randn(*dims)*np.sqrt(2/(np.sum(dims)))

initializer_handler = {
    'HeNormal' : HeNormal(),
    'XavieNormal' : XavieNormal()
}

