########################################################
# module: tfl_image_convnets.py
# authors: vladimir kulyukin
# descrption: starter code for image ConvNets for Project 1
# to install tflearn to go http://tflearn.org/installation/
########################################################

import pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# we need this to load the pickled data into Python.


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


# Paths to all datasets. Change accordingly.
PATH = '/home/fox/Documents/Intelligent Systems/project 1/datasets/'
BEE1_path = PATH + 'BEE1/'
BEE2_1S_path = PATH + 'BEE2_1S/'
BEE4_path = PATH + 'BEE4/'

# let's load BEE1
base_path = BEE1_path
print('loading datasets from {}...'.format(base_path))
BEE1_train_X = load(base_path + 'train_X.pck')
BEE1_train_Y = load(base_path + 'train_Y.pck')
BEE1_test_X = load(base_path + 'test_X.pck')
BEE1_test_Y = load(base_path + 'test_Y.pck')
BEE1_valid_X = load(base_path + 'valid_X.pck')
BEE1_valid_Y = load(base_path + 'valid_Y.pck')
print(BEE1_train_X.shape)
print(BEE1_train_Y.shape)
print(BEE1_test_X.shape)
print(BEE1_test_Y.shape)
print(BEE1_valid_X.shape)
print(BEE1_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BEE1_train_X = BEE1_train_X.reshape([-1, 64, 64, 3])
BEE1_test_X = BEE1_test_X.reshape([-1, 64, 64, 3])

# to make sure that the dimensions of the
# examples and targets are the same.
assert BEE1_train_X.shape[0] == BEE1_train_Y.shape[0]
assert BEE1_test_X.shape[0] == BEE1_test_Y.shape[0]
assert BEE1_valid_X.shape[0] == BEE1_valid_Y.shape[0]

# let's load BEE2_1S
base_path = BEE2_1S_path
print('loading datasets from {}...'.format(base_path))
BEE2_1S_train_X = load(base_path + 'train_X.pck')
BEE2_1S_train_Y = load(base_path + 'train_Y.pck')
BEE2_1S_test_X = load(base_path + 'test_X.pck')
BEE2_1S_test_Y = load(base_path + 'test_Y.pck')
BEE2_1S_valid_X = load(base_path + 'valid_X.pck')
BEE2_1S_valid_Y = load(base_path + 'valid_Y.pck')
print(BEE2_1S_train_X.shape)
print(BEE2_1S_train_Y.shape)
print(BEE2_1S_test_X.shape)
print(BEE2_1S_test_Y.shape)
print(BEE2_1S_valid_X.shape)
print(BEE2_1S_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BEE2_1S_train_X = BEE2_1S_train_X.reshape([-1, 64, 64, 3])
BEE2_1S_test_X = BEE2_1S_test_X.reshape([-1, 64, 64, 3])

assert BEE2_1S_train_X.shape[0] == BEE2_1S_train_Y.shape[0]
assert BEE2_1S_test_X.shape[0] == BEE2_1S_test_Y.shape[0]
assert BEE2_1S_valid_X.shape[0] == BEE2_1S_valid_Y.shape[0]

# let's load BEE4
base_path = BEE4_path
print('loading datasets from {}...'.format(base_path))
BEE4_train_X = load(base_path + 'train_X.pck')
BEE4_train_Y = load(base_path + 'train_Y.pck')
BEE4_test_X = load(base_path + 'test_X.pck')
BEE4_test_Y = load(base_path + 'test_Y.pck')
BEE4_valid_X = load(base_path + 'valid_X.pck')
BEE4_valid_Y = load(base_path + 'valid_Y.pck')
print(BEE4_train_X.shape)
print(BEE4_train_Y.shape)
print(BEE4_test_X.shape)
print(BEE4_test_Y.shape)
print(BEE4_valid_X.shape)
print(BEE4_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BEE4_train_X = BEE4_train_X.reshape([-1, 64, 64, 3])
BEE4_test_X = BEE4_test_X.reshape([-1, 64, 64, 3])

assert BEE4_train_X.shape[0] == BEE4_train_Y.shape[0]
assert BEE4_test_X.shape[0] == BEE4_test_Y.shape[0]
assert BEE4_valid_X.shape[0] == BEE4_valid_Y.shape[0]

# here's an example of how to make an ConvNet with tflearn.


def example_layers():
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    return fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')


def layers_1conv_1fc():
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer, nb_filter=10,
                         filter_size=5,
                         activation='relu',
                         name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_1')

    return fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='output_layer')


def layers_2conv_1fc():
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer, nb_filter=10,
                         filter_size=5,
                         activation='sigmoid',
                         name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')

    conv_layer_2 = conv_2d(pool_layer_1, nb_filter=20,
                         filter_size=5,
                         activation='relu',
                         name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer_2, 80,
                                 activation='relu',
                                 name='fc_layer_1')

    return fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='output_layer')


def layers_2conv_2fc():
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer, nb_filter=10,
                         filter_size=5,
                         activation='sigmoid',
                         name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')

    conv_layer_2 = conv_2d(pool_layer_1, nb_filter=20,
                         filter_size=5,
                         activation='relu',
                         name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer_2, 80,
                                 activation='relu',
                                 name='fc_layer_1')

    fc_layer_2 = fully_connected(fc_layer_1, 100,
                                 activation='relu',
                                 name='fc_layer_2')

    return fully_connected(fc_layer_2, 2,
                                 activation='softmax',
                                 name='output_layer')


def layers_2conv_2fc_small_kernel():
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer, nb_filter=10,
                         filter_size=3,
                         activation='sigmoid',
                         name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')

    conv_layer_2 = conv_2d(pool_layer_1, nb_filter=20,
                         filter_size=3,
                         activation='relu',
                         name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer_2, 80,
                                 activation='relu',
                                 name='fc_layer_1')

    fc_layer_2 = fully_connected(fc_layer_1, 100,
                                 activation='relu',
                                 name='fc_layer_2')

    return fully_connected(fc_layer_2, 2,
                                 activation='softmax',
                                 name='output_layer')


def layers_2conv_2fc_large_kernel():
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer, nb_filter=10,
                         filter_size=5,
                         activation='sigmoid',
                         name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')

    conv_layer_2 = conv_2d(pool_layer_1, nb_filter=20,
                         filter_size=7,
                         activation='relu',
                         name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer_2, 80,
                                 activation='relu',
                                 name='fc_layer_1')

    fc_layer_2 = fully_connected(fc_layer_1, 100,
                                 activation='relu',
                                 name='fc_layer_2')

    return fully_connected(fc_layer_2, 2,
                                 activation='softmax',
                                 name='output_layer')


def layers_2conv_2fc_dropout():
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer, nb_filter=10,
                         filter_size=5,
                         activation='sigmoid',
                         name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')

    conv_layer_2 = conv_2d(pool_layer_1, nb_filter=20,
                         filter_size=5,
                         activation='relu',
                         name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer_2, 80,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_1 = dropout(fc_layer_1, 0.5)

    fc_layer_2 = fully_connected(fc_layer_1, 100,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_2 = dropout(fc_layer_2, 0.5)

    return fully_connected(fc_layer_2, 2,
                                 activation='softmax',
                                 name='output_layer')


def best_convnet_layers():
    return layers_2conv_2fc_small_kernel()


def make_convnet_model(layers):
    network = regression(layers, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
    model = tflearn.DNN(network)
    return model


def load_image_convnet_model(model_path):
    tf.compat.v1.reset_default_graph()
    layers = best_convnet_layers()
    model = tflearn.DNN(layers)
    model.load(model_path, weights_only=True)
    return model


def test_tfl_image_convnet_model(network_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = network_model.predict(validX[i].reshape([-1, 64, 64, 3]))
        results.append(np.argmax(prediction, axis=1)[0] ==
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True))) / float(len(results))

# train a tfl convnet model on train_X, train_Y, test_X, test_Y.


def train_tfl_image_convnet_model(model, train_X, train_Y, test_X, test_Y, num_epochs=2, batch_size=10):
    tf.compat.v1.reset_default_graph()
    model.fit(train_X, train_Y, n_epoch=num_epochs,
              shuffle=True,
              validation_set=(test_X, test_Y),
              show_metric=True,
              batch_size=batch_size,
              run_id='image_cn_model')

# validating is testing on valid_X and valid_Y.


def validate_tfl_image_convnet_model(model, valid_X, valid_Y):
    return test_tfl_image_convnet_model(model, valid_X, valid_Y)
