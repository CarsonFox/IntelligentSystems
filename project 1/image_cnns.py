#!/bin/python

from os import path
from functools import reduce

import numpy as np
from tfl_image_convnets import *

def big_train():
    ann = make_convnet_model(best_convnet_layers())

    datasets = [
        ((BEE1_train_X, BEE1_train_Y), (BEE1_test_X, BEE1_test_Y)),
        ((BEE2_1S_train_X, BEE2_1S_train_Y), (BEE2_1S_test_X, BEE2_1S_test_Y)),
        ((BEE4_train_X, BEE4_train_Y), (BEE4_test_X, BEE4_test_Y)),
    ]

    for (train_X, train_Y), (test_X, test_Y) in datasets:
        train_tfl_image_convnet_model(ann,
                                  train_X, train_Y,
                                  test_X, test_Y,
                                  num_epochs=10)

    ann.save('models/img_cn.tfl')

def benchmark(name, layers):
    epochs = 10
    ann = make_convnet_model(layers)

    test_accuracies = []
    valid_accuracies = []
    for _ in range(epochs // 2):
        train_tfl_image_convnet_model(ann,
                                  BEE1_train_X, BEE1_train_Y,
                                  BEE1_test_X, BEE1_test_Y)

        test_accuracies.append(
            test_tfl_image_convnet_model(ann, BEE1_test_X, BEE1_test_Y))
        valid_accuracies.append(
            validate_tfl_image_convnet_model(ann, BEE1_valid_X, BEE1_valid_Y))

        print(test_accuracies, valid_accuracies)

    save_path = path.join('./models/', name)
    print(f'Saving ann to {save_path}')
    ann.save(save_path)
    return (name, test_accuracies, valid_accuracies)

if __name__ == "__main__":
    big_train()
