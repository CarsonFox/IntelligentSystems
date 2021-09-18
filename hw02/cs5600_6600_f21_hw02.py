#!/usr/bin/python

#########################################
# module: cs5600_6600_f21_hw02.py
# YOUR NAME
# YOUR A#
#########################################

############################################
# Neural networks were trained with 1,000
# Iterations. The structures matched those
# found in the unit tests; binary operators
# used 2x3x1 and 2x3x3x1, not used 1x2x1 and
# 1x2x2x1, and the boolean expression used
# 4x2x1 and 4x2x2x1.
############################################

import numpy as np
import pickle
from itertools import accumulate
from cs5600_6600_f21_hw02_data import *

# sigmoid function and its derivative.
# you'll use them in the training and fitting
# functions below.
def sigmoidf(x):
    return 1 / (1 + np.exp(-x))

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

def feed_forward(X, nn):
    return list(
        accumulate(nn, lambda x, y: sigmoidf(x @ y), initial=X)
    )

def backprop(activations, y, X, nn):
    yHat = activations[-1]
    yHatErr = y - yHat
    delta = yHatErr * sigmoidf_prime(yHat)

    for i in range(1, len(nn)):
        error = delta @ nn[-i].T
        nn[-i] += activations[-i - 1].T @ delta
        delta = error * sigmoidf_prime(activations[-i - 1])

    nn[0] += X.T @ delta

## Training 3-layer neural net.
## X is the matrix of inputs
## y is the matrix of ground truths.
## build is a nn builder function.
def train_3_layer_nn(numIters, X, y, build):
    nn = build()
    for _ in range(numIters):
        activations = feed_forward(X, nn)
        backprop(activations, y, X, nn)
    return nn

def train_4_layer_nn(numIters, X, y, build):
    #Same implementation
    return train_3_layer_nn(numIters, X, y, build)

def fit_3_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    threshold = np.frompyfunc(lambda x: 1 if x > thresh else 0, 1, 1)
    yHat = feed_forward(x, wmats)[-1]
    return threshold(yHat) if thresh_flag else yHat

def fit_4_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    #Same implementation
    return fit_3_layer_nn(x, wmats, thresh=thresh, thresh_flag=thresh_flag)
