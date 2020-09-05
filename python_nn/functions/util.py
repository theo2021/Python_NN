import numpy as np

def val2onehot(val_array):
    unique = np.unique(val_array)
    u_d = dict(zip(unique, np.arange(len(unique))))
    labels = np.zeros((len(val_array), len(unique)))
    for ind,lbl in enumerate(val_array):
        labels[ind,u_d[lbl]] = 1
    return labels
