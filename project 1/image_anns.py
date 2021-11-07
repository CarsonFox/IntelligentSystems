#!/bin/python

import numpy as np
from tfl_image_anns import *

if __name__ == "__main__":
    epochs = 40

    test_accuracies = []
    valid_accuracies = []

    for i in range(epochs//2):
        layers = layers_16()
        ann = make_image_ann(layers)
        train_tfl_image_ann_model(ann,
                                  BEE1_gray_train_X, BEE1_gray_train_Y,
                                  BEE1_gray_test_X, BEE1_gray_test_Y)

        test_accuracies.append(
            test_tfl_image_ann_model(ann, BEE1_gray_test_X, BEE1_gray_test_Y))
        valid_accuracies.append(
            validate_tfl_image_ann_model(ann, BEE1_gray_valid_X, BEE1_gray_valid_Y))

    print(','.join(test_accuracies))
    print(','.join(valid_accuracies))
