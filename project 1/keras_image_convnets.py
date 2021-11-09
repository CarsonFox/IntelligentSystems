########################################################
# module: keras_image_convnets.py
# authors: vladimir kulyukin
# descrption: starter code for keras image classification
#             nets for project 1
########################################################

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle

from keras_load_data import dataset, num_classes, train_X, train_Y, test_X, test_Y, valid_X, valid_Y

print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

tf.random.set_seed(1234)

# change as necessary; this is the path where ConvNet models
# are persisted.
MODEL_PATH = '/home/fox/Documents/Intelligent Systems/project 1/keras_models'
MODEL_NAME = 'keras_img_model'

# Define a simple sequential model with Keras


def example_create_keras_model(learning_rate=0.01, weight_decay=0.001, dropout=0.2):
    model = keras.models.Sequential([
        # 32 filters, each of which is 3x3, padding="same" means that the output
        # is of the same size of the input, which in this case will be 64x64.
        keras.layers.Conv2D(32, (3, 3), padding="same",
                            activation='relu', input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(
            128, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_keras_model_2hl(learning_rate=0.01, weight_decay=0.001, dropout=0.2):
    model = keras.models.Sequential([
        # 32 filters, each of which is 3x3, padding="same" means that the output
        # is of the same size of the input, which in this case will be 64x64.
        keras.layers.Conv2D(32, (3, 3), padding="same",
                            activation='relu', input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(dropout),

        keras.layers.Dense(
            128, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)),

        # Second hidden layer
        keras.layers.Dense(
            64, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)),

        keras.layers.Dense(num_classes, activation='softmax')
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_keras_model_2cl(learning_rate=0.01, weight_decay=0.001, dropout=0.2):
    model = keras.models.Sequential([
        # 32 filters, each of which is 3x3, padding="same" means that the output
        # is of the same size of the input, which in this case will be 64x64.
        keras.layers.Conv2D(32, (3, 3), padding="same",
                            activation='relu', input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), padding="same",
                            activation='relu', input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(dropout),

        keras.layers.Dense(
            128, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)),

        keras.layers.Dense(num_classes, activation='softmax')
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# def create_keras_model_2cl_2hl(learning_rate=0.01, weight_decay=0.001, dropout=0.2):
def create_keras_model(learning_rate=0.01, weight_decay=0.001, dropout=0.2):
    model = keras.models.Sequential([
        # 32 filters, each of which is 3x3, padding="same" means that the output
        # is of the same size of the input, which in this case will be 64x64.
        keras.layers.Conv2D(32, (3, 3), padding="same",
                            activation='relu', input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), padding="same",
                            activation='relu', input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(dropout),

        keras.layers.Dense(
            128, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)),

        # Second hidden layer
        keras.layers.Dense(
            64, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)),

        keras.layers.Dense(num_classes, activation='softmax')
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_keras_model(learning_rate=0.01, weight_decay=0.001, dropout=0.2, epochs=80):
    # Create a basic model instance
    model = create_keras_model(learning_rate, weight_decay, dropout)

    # Display the model's architecture
    model.summary()

    checkpoint_filepath = MODEL_PATH + '/' + MODEL_NAME
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                save_weights_only=True,
                                                                monitor='val_loss',
                                                                mode='min',
                                                                save_best_only=True)
    # Fit the model
    model.fit(train_X, train_Y,
              validation_data=(test_X, test_Y),
              batch_size=64,
              epochs=epochs,
              callbacks=[model_checkpoint_callback])
    model.save(MODEL_PATH + '/' + 'keras_img_model' + '.h5')


def load_keras_model():
    km = create_keras_model(
        learning_rate=0.01, weight_decay=0.001, dropout=0.2)
    km.load_weights(MODEL_PATH + '/' + MODEL_NAME + '.h5')
    return km


def load_keras_model_from_file(path):
    km = create_keras_model(
        learning_rate=0.01, weight_decay=0.001, dropout=0.2)
    km.load_weights(path)
    return km


def evaluate_keras_model(model):
    train_loss, train_acc = model.evaluate(train_X, train_Y, verbose=0)
    print("Training Accuracy: {:5.2f}%".format(100 * train_acc))

    test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=0)
    print("Testing Accuracy: {:5.2f}%".format(100 * test_acc))

    valid_loss, valid_acc = model.evaluate(valid_X,  valid_Y, verbose=0)
    print("Validation Accuracy: {:5.2f}%".format(100 * valid_acc))

    return [train_loss, train_acc, test_loss, test_acc, valid_loss, valid_acc]

# ========================================================================


if __name__ == '__main__':
    pass
