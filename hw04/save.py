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

#nn = ann([784, 11, 10])
#stats = nn.mini_batch_sgd(train_d, 30, 10, .7, lmbda=.6, evaluation_data=valid_d,
#                  monitor_training_cost=True, monitor_training_accuracy=True,
#                  monitor_evaluation_cost=True, monitor_evaluation_accuracy=True)
#plot_costs(stats[0], stats[2], 30)
#plot_accuracies(stats[1], stats[3], 30)
#nn.save('net1.json')

nn = ann([784, 10, 11, 10])
stats = nn.mini_batch_sgd(train_d, 30, 10, .5, lmbda=.2, evaluation_data=valid_d,
                  monitor_training_cost=True, monitor_training_accuracy=True,
                  monitor_evaluation_cost=True, monitor_evaluation_accuracy=True)
plot_costs(stats[0], stats[2], 30)
plot_accuracies(stats[1], stats[3], 30)
nn.save('net2.json')
