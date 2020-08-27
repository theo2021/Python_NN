import numpy as np
#from python_nn.layers import *
from .functions.loss_functions import los_handler, reg_handler
from .functions.activation_functions import activation_handler
from .optimizers import *
from .initializers import *
from tqdm import tqdm

def onehot2char(val_array, dictionary):
    char_arr = np.array(list(dictionary.keys()))[np.argmax(val_array, axis=1)]
    return ''.join(char_arr)

class Sequential:

    def __init__(self, input_dim=None, dim_2d=None):
        self.cur_dim = input_dim
        self.cur_2d = dim_2d
        if input_dim is None:
            self.cur_dim = dim_2d[0] * dim_2d[1]
        self.layers = []
        self.compile()
        self.store = {}
        self.ram ={}

    def add_layer(self, layer_type, **args):
        self.layers.append(layer_type(previous_layer=self.cur_dim, input_dims=self.cur_2d, **args))
        self.cur_dim = self.layers[-1].feed(np.ones((1, self.cur_dim)))[1].shape[1]
        self.cur_2d = self.layers[-1].output_dims()
        self.layers[-1].who_am_i = len(self.layers)-1

    def feed_forward(self, input, only_sol=0):
        layer_pairs = [(0, input)]
        for layer in self.layers:
            layer_pairs.append(layer.feed(layer_pairs[-1][1]))
        if only_sol:
            return layer_pairs[-1][1]
        return layer_pairs

    def compile(self, loss_func='mse', loss_weights=None, regularizer={'name': 'none', 'args': {}}, optimizer={'name': 'SGD', 'args': {'learning_rate': 0.001, 'momentum': 0}}):
        self.loss = los_handler(loss_func, loss_weights)
        self.update_handler = {}
        self.update_handler['reg'] = reg_handler(regularizer['name'], **regularizer['args'])
        optimizer['args']['layers'] = len(self.layers)
        self.update_handler['optimizer'] =  opt_handler(optimizer['name'], **optimizer['args'])
        
    def regularization_loss(self):
        loss = 0
        for layer in self.layers:
            for param in layer.storage['parameters'].values():
                loss += self.update_handler['reg'].cost(param)
        return loss

    def get_update_params(self):
        return self.update_handler

    def back_prob(self, signal_pairs, targets):
        net_output = signal_pairs[-1][1] # last signal
        # signal_pairs layers + 1
        update_params = self.get_update_params()
        delta = self.loss.dev_func(targets, net_output)
        for i in range(len(self.layers)-1, -1, -1):
            delta = self.layers[i].update_return(delta, signal_pairs[i+1][0], signal_pairs[i][1], update_params)
    #
    # def back_prob_test(self, signal_pairs, targets, loop):
    #     net_output = signal_pairs[-1][1] # last signal
    #     # signal_pairs layers + 1
    #     update_params = self.get_update_params()
    #     delta = self.loss.dev_func(targets, net_output)
    #     delta = self.layers[2].update_return(delta, signal_pairs[2 + 1][0], signal_pairs[2][1], update_params)
    #     delta = self.layers[1].update_return(delta, signal_pairs[1 + 1][0], signal_pairs[1][1], update_params)
    #     self.layers[0].update_return(delta, signal_pairs[1][0], signal_pairs[0][1], update_params)

    def evaluate(self, test_set, labels):
        out = self.feed_forward(test_set)[-1][1]
        acc = np.sum((np.argmax(out, axis=1) == np.argmax(labels, axis=1))*1)/labels.shape[0]
        loss = self.loss.func(labels, out)
        return acc, loss, loss + self.regularization_loss()

    # def monitor(self, label, inputs, targets):
    #     self.store[label] = {}
    #     self.store[label]['inp'] = inputs
    #     self.store[label]['out'] = targets
    #     self.store[label]['rec'] = []

    # def clear_monitor(self):
    #     self.store = {}

    # def get_monitor(self, label):
    #     return np.array(self.store[label]['rec'])

    # def run_monitor(self):
    #     for label in self.store.keys():
    #         self.store[label]['rec'].append(list(self.evaluate(self.store[label]['inp'], self.store[label]['out'])))

    def get_all_params(self):
        tmp = np.empty(0)
        for layer in self.layers:
            tmp = np.hstack((tmp, layer.get_params_vec()))
        return tmp

    def get_dev_params(self):
        tmp = np.empty(0)
        for layer in self.layers:
            tmp = np.hstack((tmp, layer.get_dev_vec()))
        return tmp

    def update_all_params(self, vec):
        pos = 0
        for i in range(len(self.layers)):
            parameters = self.layers[i].params
            self.layers[i].edit_params(vec[pos:pos + parameters])
            pos += parameters

    def compute_numerical_gradient(self, inputs, targets, acc=10**(-10)):
        cur_params = self.get_all_params()
        derivative = np.zeros(cur_params.shape)
        for i in range(len(cur_params)):
            test = np.array(cur_params)
            test[i] += acc
            self.update_all_params(test)
            derivative[i] = self.evaluate(inputs, targets)[1]/(2*acc)
            test[i] -= 2*acc
            self.update_all_params(test)
            derivative[i] -= self.evaluate(inputs, targets)[1] / (2*acc)
        self.update_all_params(cur_params)
        return derivative

    def train(self, inputs, targets, epochs, batch_size=1, dicti=None, callbacks=[]):
        
        loops = int(len(inputs) / batch_size)
        additional_loop = (batch_size * loops < inputs.shape[0])*1
        self.store['training_session'] = {'epochs' : epochs, 'iterations' : epochs*(loops + additional_loop), 'inputs': inputs, 'targets': targets}
        iteration = 0
        for epoch in range(0, epochs):
            prev = 0
            for loop in range(loops + additional_loop):
                # Running all callbacks
                self.store['training_session']['iteration'] = iteration
                self.store['training_session']['epoch'] = epoch
                for i in range(len(callbacks)):
                    callbacks[i](self)
                # getting inputs and targets from batch 
                inp_batch = self.store['training_session']['inputs'][prev: prev + batch_size]
                trg_batch = self.store['training_session']['targets'][prev: prev + batch_size]
                prev += batch_size # the new batch position
                pairs = self.feed_forward(inp_batch)
                loss = self.loss.func(trg_batch, pairs[-1][1]) # for smooth loss
                loss *= batch_size
                self.store['training_session']['loss'] = loss
                self.back_prob(pairs, trg_batch)
                iteration += 1
                

    def syntesize(self, inputs, length=1):
        # Only availiable in RNN
        # 2d matrix of inputs have to change RNN to remember
        def sample(vec):
            vec = vec**(2)
            vec = vec/np.sum(vec)
            maxim = np.cumsum(vec)
            rand = np.random.rand()
            out = np.zeros_like(vec)
            out[np.where(maxim-rand>0)[0][0]] = 1
            return out
        # for layer_index in range(len(self.layers)):
        #         if self.layers[layer_index].recognizer == 'Recurent':
        #             self.layers[layer_index].h_init()
        output = sample(self.feed_forward(inputs, only_sol=1)[-1]).reshape(1,-1)
        for i in range(length):
            output = np.vstack((output, sample(self.feed_forward(output[-1].reshape(1,-1), only_sol=1)[0])))
        return output

    # def train_test(self, inputs, targets, epochs, batch_size=1):
    #     # just implemented the slow version to measure time
    #     loops = int(len(inputs) / batch_size)
    #     tmp = np.random.permutation(inputs.shape[0])
    #     inputs = inputs[tmp]
    #     targets = targets[tmp]
    #     for epoch in tqdm(range(0, epochs)):
    #         # shuffle
    #         self.run_monitor()
    #         prev = 0
    #         for loop in range(loops):
    #             inp_batch = inputs[prev: prev + batch_size]
    #             trg_batch = targets[prev: prev + batch_size]
    #             prev += batch_size
    #             pairs = self.feed_forward(inp_batch)
    #             self.back_prob_test(pairs, trg_batch, loop)
    #         if batch_size * loops < inputs.shape[0]:
    #             inp_batch = inputs[batch_size * loops:]
    #             trg_batch = targets[batch_size * loops:]
    #             pairs = self.feed_forward(inp_batch)
    #             self.back_prob_test(pairs, trg_batch, loops+1)
