#!/usr/bin/python

#########################################
# module: cs5600_6600_f21_hw02.py
# YOUR NAME
# YOUR A#
#########################################

import numpy as np
import pickle
from cs5600_6600_f21_hw02_data import *

# sigmoid function and its derivative.
# you'll use them in the training and fitting
# functions below.
def sigmoidf(x):
    return 1 / (1 + exp(-x))

def sigmoidf_prime(x):
    #approximate
    return x * (1 - x)

# persists object obj to a file with pickle.dump()
def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

# restores the object from a file with pickle.load()
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def build_nn_wmats(mat_dims):
    pairs = zip(mat_dims[:-1], mat_dims[1:])
    return [np.random.rand(m, n) for (m, n) in pairs]

def build_231_nn():
    return build_nn_wmats((2, 3, 1))

def build_2331_nn():
    return build_nn_wmats((2, 3, 3, 1))

def build_221_nn():
    return build_nn_wmats((2, 2, 1))

def build_838_nn():
    return build_nn_wmats((8, 3, 8))

def build_949_nn():
    return build_nn_wmats((9, 4, 9))

def build_4221_nn():
    return build_nn_wmats((4, 2, 2, 1))

def build_421_nn():
    return build_nn_wmats((4, 2, 1))

def build_121_nn():
    return build_nn_wmats((1, 2, 1))

def build_1221_nn():
    return build_nn_wmats((1, 2, 2, 1))

## Training 3-layer neural net.
## X is the matrix of inputs
## y is the matrix of ground truths.
## build is a nn builder function.
def train_3_layer_nn(numIters, X, y, build):
    ## your code here
    pass

def train_4_layer_nn(numIters, X, y, build):
    ## your code here
    pass

def fit_3_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    ## your code here
    pass

def fit_4_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    ## your code here
    pass

