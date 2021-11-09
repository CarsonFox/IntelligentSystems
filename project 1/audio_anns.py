#!/bin/python

from os import path

import numpy as np
from tfl_audio_anns import *

def benchmark(name, layers):
    epochs = 40
    ann = make_audio_ann_model(layers)

    test_accuracies = []
    valid_accuracies = []
    for _ in range(epochs // 2):
        train_tfl_audio_ann_model(ann,
                                  BUZZ1_train_X, BUZZ1_train_Y,
                                  BUZZ1_test_X, BUZZ1_test_Y)

        test_accuracies.append(
            test_tfl_audio_ann_model(ann, BUZZ1_test_X, BUZZ1_test_Y))
        valid_accuracies.append(
            validate_tfl_audio_ann_model(ann, BUZZ1_valid_X, BUZZ1_valid_Y))

        print(test_accuracies, valid_accuracies)

    save_path = path.join('./models/', name)
    print(f'Saving ann to {save_path}')
    ann.save(save_path)
    return (name, test_accuracies, valid_accuracies)

if __name__ == "__main__":
    with open('output', 'w') as f:
        networks = {
            '16': layers_16,
            '16x16': layers_16x16,
            '16x16_dropout': layers_16x16_dropout,
            '32': layers_32,
            '32x32': layers_32x32,
            '32x32_dropout': layers_32x32_dropout,
            '32x32x32': layers_32x32x32,
            '64x64': layers_64x64,
            '64x64x64': layers_64x64x64,
        }

        benchmarks = [benchmark(name, layers()) for name, layers in networks.items()]

        for name, test, valid in benchmarks:
            print(f'{name},')
            print(','.join(map(str, test)))
            print(','.join(map(str, valid)))
            f.write(f'{name},')
            f.write(','.join(map(str, test)))
            f.write(','.join(map(str, valid)))
