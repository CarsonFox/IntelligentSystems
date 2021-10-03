import json
import random
import sys
import numpy as np
import unittest
from itertools import product

from cs5600_6600_f21_hw04 import *
from mnist_loader import load_data_wrapper

# change this directory accordingly.
DIR_PATH = '/home/fox/Documents/Intelligent Systems/hw04/'

train_d, valid_d, test_d = load_data_wrapper()

lmbda = list(map(lambda x: x/10, range(0, 8, 2)))
eta = list(map(lambda x: x/10, range(1, 9, 2)))

def find_best_size(training_stats):
    final_accuracies = { n: stats[3][-1] for n, stats in training_stats.items() }
    best = max(final_accuracies, key=final_accuracies.get)
    return best, final_accuracies[best]

def find_best_params(stats_fn):
    stats = { f'lambda = {l}, eta = {e}': find_best_size(stats_fn(l, e))
             for l, e in product(lmbda, eta) }
    maxes = { f'{size} network with {params}': best
             for params, (size, best) in stats.items() }
    best = max(maxes, key=maxes.get)
    return best, maxes[best]

d1 = lambda l, e: collect_1_hidden_layer_net_stats(10, 11, CrossEntropyCost, 5, 10, e, l, train_d, test_d)
d2 = lambda l, e: collect_2_hidden_layer_net_stats(10, 11, CrossEntropyCost, 5, 10, e, l, train_d, test_d)
d3 = lambda l, e: collect_3_hidden_layer_net_stats(10, 11, CrossEntropyCost, 5, 10, e, l, train_d, test_d)

print('Best 1-layer size: ', find_best_params(d1))
print('Best 2-layer size: ', find_best_params(d2))
print('Best 3-layer size: ', find_best_params(d3))
