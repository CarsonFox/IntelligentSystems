#!/bin/python

from os import path

import numpy as np
from tfl_audio_convnets import *

def benchmark(name, layers):
    epochs = 10
    ann = make_audio_convnet_model(layers)

    test_accuracies = []
    valid_accuracies = []
    for _ in range(epochs // 2):
        train_tfl_audio_convnet_model(ann,
                                  BUZZ1_train_X, BUZZ1_train_Y,
                                  BUZZ1_test_X, BUZZ1_test_Y)

        test_accuracies.append(
            test_tfl_audio_convnet_model(ann, BUZZ1_test_X, BUZZ1_test_Y))
        valid_accuracies.append(
            validate_tfl_audio_convnet_model(ann, BUZZ1_valid_X, BUZZ1_valid_Y))

        print(test_accuracies, valid_accuracies)

    save_path = path.join('./models/', name)
    print(f'Saving ann to {save_path}')
    ann.save(save_path)
    return (name, test_accuracies, valid_accuracies)

if __name__ == "__main__":
    with open('output', 'w') as f:
        networks = {
            '1x1': layers_1conv_1fc,
            '2x1': layers_2conv_1fc,
            '2x2': layers_2conv_2fc,
            '2x2_dropout': layers_2conv_2fc_dropout,
            '2x2_large': layers_2conv_2fc_large_kernel,
            '2x2_small': layers_2conv_2fc_small_kernel,
        }

        benchmarks = [benchmark(name, layers()) for name, layers in networks.items()]

        for name, test, valid in benchmarks:
            print(f'{name},')
            print(','.join(map(str, test)))
            print(','.join(map(str, valid)))
            f.write(f'{name},')
            f.write(','.join(map(str, test)))
            f.write(','.join(map(str, valid)))
