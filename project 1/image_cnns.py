#!/bin/python

from os import path

import numpy as np
from tfl_image_convnets import *

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
    with open('output', 'w') as f:
        networks = {
            #'1x1': layers_1conv_1fc,
            #'2x1': layers_2conv_1fc,
            #'2x2': layers_2conv_2fc,
            #'dropout': layers_2conv_2fc_dropout,
            #'small': layers_2conv_2fc_small_kernel,
            #'large': layers_2conv_2fc_large_kernel,
            'img_cnn.tfl': layers_2conv_2fc_large_kernel,
        }

        benchmarks = [benchmark(name, layers()) for name, layers in networks.items()]

        for name, test, valid in benchmarks:
            print(f'{name},')
            print(','.join(map(str, test)))
            print(','.join(map(str, valid)))
