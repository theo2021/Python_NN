import numpy as np
import pandas as pd

class Callback:

    def __init__(self, epoch=None, iteration=None):
        if (epoch is None) and (iteration is None):
            #raise ValueError('Frequency of the monitor should include iterations or frequency.')
            epoch = 1
            iteration = np.inf
        elif epoch is None:
            epoch = np.inf
        elif iteration is None:
            iteration = np.inf
        self.epoch = epoch
        self.prev_epoch = -1
        self.iteration = iteration

    def __call__(self, model):
        if self. epoch is np.inf:
            if model.store['training_session']['iteration'] % self.iteration == 0:
                self.run(model)
        else:
            if model.store['training_session']['epoch'] % self.epoch == 0 and self.prev_epoch != model.store['training_session']['epoch']:
                self.run(model)
                self.prev_epoch = model.store['training_session']['epoch']

    def run(self, model):
        raise NotImplementedError


class Monitor(Callback):

    def __init__(self, samples, targets, epoch=None, iteration=None):
        super().__init__(epoch, iteration)
        self.info = ['epoch', 'iteration', 'accuracy', 'loss', 'total loss']
        self.data = np.empty((0,5))
        self.samples = samples
        self.targets = targets

    def run(self, model):
        epoch = model.store['training_session']['epoch'] 
        iteration = model.store['training_session']['iteration']
        performance = model.evaluate(self.samples, self.targets)
        self.data = np.vstack((self.data, [epoch, iteration, *performance]))

    
    def clear(self):
        self.data = np.empty((0,5))

    def pandas(self):
        return pd.DataFrame(self.data, columns=self.info)

class Triangular(Callback):

    def __init__(self, eta_min, eta_max, ns):
        super().__init__(None, 1) # running the run function in each iteration
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.ns = ns

    def run(self, model):
        iteration = model.store['training_session']['iteration']
        pos_in_loop = iteration % (2*self.ns)
        if pos_in_loop <= self.ns:
            new_step = self.eta_min + pos_in_loop / self.ns * (self.eta_max - self.eta_min)
        else:
            new_step = self.eta_max - (pos_in_loop - self.ns) / self.ns * (self.eta_max - self.eta_min)
        model.update_handler['optimizer'].lr = new_step