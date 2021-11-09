#!/bin/python

from os import path

import numpy as np
from tfl_audio_anns import *

def big_train():
    ann = make_audio_ann_model(best_layers())

    datasets = [
        ((BUZZ1_train_X, BUZZ1_train_Y),
         (BUZZ1_test_X, BUZZ1_test_Y)),
        ((BUZZ2_train_X, BUZZ2_train_Y),
         (BUZZ2_test_X, BUZZ2_test_Y)),
        ((BUZZ3_train_X, BUZZ3_train_Y),
         (BUZZ3_test_X, BUZZ3_test_Y)),
    ]

    for (train_X, train_Y), (test_X, test_Y) in datasets:
        train_tfl_audio_ann_model(ann,
                                  train_X, train_Y,
                                  test_X, test_Y,
                                  num_epochs=20)

    ann.save('models/aud_ann.tfl')

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
    big_train()
