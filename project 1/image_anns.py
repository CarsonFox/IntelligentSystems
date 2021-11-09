#!/bin/python

from os import path

import numpy as np
from tfl_image_anns import *

def big_train():
    ann = make_image_ann(best_ann_model())

    datasets = [
        ((BEE1_gray_train_X, BEE1_gray_train_Y),
         (BEE1_gray_test_X, BEE1_gray_test_Y)),
        ((BEE2_1S_gray_train_X, BEE2_1S_gray_train_Y),
         (BEE2_1S_gray_test_X, BEE2_1S_gray_test_Y)),
        ((BEE4_gray_train_X, BEE4_gray_train_Y),
         (BEE4_gray_test_X, BEE4_gray_test_Y)),
    ]

    for (train_X, train_Y), (test_X, test_Y) in datasets:
        train_tfl_image_ann_model(ann,
                                  train_X, train_Y,
                                  test_X, test_Y,
                                  num_epochs=20)

    ann.save('models/img_ann.tfl')

def benchmark(name, layers):
    epochs = 80
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
    big_train()
