#/usr/bin/python

from ann import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

####################################
# CS5600/6600: F21: HW04
# Your Name
# Your A#
# Write your code at the end of
# this file in the provided
# function stubs.
#####################################

#### auxiliary functions
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of ann.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = ann(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### plotting costs and accuracies
def plot_costs(eval_costs, train_costs, num_epochs):
    plt.title('Evaluation Cost (EC) and Training Cost (TC)')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    epochs = [i for i in range(num_epochs)]
    plt.plot(epochs, eval_costs, label='EC', c='g')
    plt.plot(epochs, train_costs, label='TC', c='b')
    plt.grid()
    plt.autoscale(tight=True)
    plt.legend(loc='best')
    plt.show()

def plot_accuracies(eval_accs, train_accs, num_epochs):
    plt.title('Evaluation Acc (EA) and Training Acc (TC)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    epochs = [i for i in range(num_epochs)]
    plt.plot(epochs, eval_accs, label='EA', c='g')
    plt.plot(epochs, train_accs, label='TA', c='b')
    plt.grid()
    plt.autoscale(tight=True)
    plt.legend(loc='best')
    plt.show()

def net_stats(layers,
                                     cost_function,
                                     num_epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    net = ann(layers, cost=cost_function)
    return net.mini_batch_sgd(train_data, num_epochs, mbs, eta, lmbda=lmbda, evaluation_data=eval_data, monitor_evaluation_accuracy=True,
                              monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)

## num_nodes -> (eval_cost, eval_acc, train_cost, train_acc)
## use this function to compute the eval_acc and min_cost.
def collect_1_hidden_layer_net_stats(*args):
    (l, u, *rest) = args
    return { n: net_stats(*([784, n, 10], *rest)) for n in range(l, u + 1) }

def collect_2_hidden_layer_net_stats(*args):
    (l, u, *rest) = args
    return { n: net_stats(*([784, n, 10], *rest)) for n in range(l, u + 1) }

def collect_3_hidden_layer_net_stats(*args):
    (l, u, *rest) = args
    return { n: net_stats(*([784, n, 10], *rest)) for n in range(l, u + 1) }
