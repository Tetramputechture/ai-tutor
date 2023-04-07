import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers, applications
import string

TOKENS = list(string.digits + '+=/') + ['START'] + ['END']

MAX_EQ_LENGTH = 23 + len('START') + len('END')


class Model:
    def create_model(self, input_shape=(100, 100, 3)):
        resnet_base = applications.resnet.ResNet50(
            include_top=False,
            input_shape=input_shape
        )
        model = models.Sequential([
            resnet_base,
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(TOKENS), activation='softmax')
        ])

        # model.compile(optimizer='adam',
        #               loss='mse',
        #               metrics=['accuracy'])
        return model
