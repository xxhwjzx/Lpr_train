# -- encoding:utf-8 --
import os
import argparse

import cv2
import numpy as np
from keras import backend as K
from keras.layers import Input, Activation, Conv2D, BatchNormalization, Lambda, MaxPooling2D,concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


def build_model(width, height, num_channels, classes):
    input_tensor = Input(name='the_input', shape=(width, height, num_channels), dtype='float32')
    x = input_tensor
    base_conv = 32

    for i in range(3):
        x = Conv2D(base_conv * (2 ** i), (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (5, 5))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    conx1 = Conv2D(256, (7, 1), padding='same')(x)
    conx1 = BatchNormalization()(conx1)
    conx2 = Conv2D(256, (5, 1), padding='same')(x)
    conx2 = BatchNormalization()(conx2)
    conx3 = Conv2D(256, (3, 1), padding='same')(x)
    conx3 = BatchNormalization()(conx3)
    conx4 = Conv2D(256, (1, 1), padding='same')(x)
    conx4 = BatchNormalization()(conx4)
    x=concatenate([conx1,conx2,conx3,conx4],axis=-1)
    x = Conv2D(1024, (1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(classes, (1, 1))(x)
    x = Activation('softmax')(x)
    model = Model(input_tensor, x, name='miniggnet')
    # y_pred = x
    # return input_tensor, y_pred
    return model
