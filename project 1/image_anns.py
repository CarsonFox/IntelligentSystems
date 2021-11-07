#!/bin/python

from os import path

import numpy as np
from tfl_image_anns import *

def benchmark(name, layers):
    epochs = 40
    ann = make_image_ann(layers)

    test_accuracies = []
    valid_accuracies = []
    for _ in range(epochs // 2):
        train_tfl_image_ann_model(ann,
                                  BEE1_gray_train_X, BEE1_gray_train_Y,
                                  BEE1_gray_test_X, BEE1_gray_test_Y)

        test_accuracies.append(
            test_tfl_image_ann_model(ann, BEE1_gray_test_X, BEE1_gray_test_Y))
        valid_accuracies.append(
            validate_tfl_image_ann_model(ann, BEE1_gray_valid_X, BEE1_gray_valid_Y))

        print(test_accuracies, valid_accuracies)

    save_path = path.join('./models/', name)
    print(f'Saving ann to {save_path}')
    ann.save(save_path)
    return (name, test_accuracies, valid_accuracies)

if __name__ == "__main__":
    with open('output', 'w') as f:
        networks = {
            '16': layers_16,
            '32': layers_32,
            '16x16': layers_16x16,
            '16x16_dropout': layers_16x16_dropout,
            '32x32': layers_32x32,
            '32x32_dropout': layers_32x32_dropout,
        }

        benchmarks = [benchmark(name, layers()) for name, layers in networks.items()]

        print(benchmarks)
        f.write(f'{benchmarks}')
