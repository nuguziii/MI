import numpy as np

def one_hot_to_index(arr):
    return np.argmax(np.array(arr), axis=1)

def index_to_one_hot(arr, nb_classes):
    return np.moveaxis(np.eye(nb_classes)[np.array(arr)], -1, 1)

