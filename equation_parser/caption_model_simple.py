import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, layers, models, optimizers, applications
import string

from .tokens import TOKENS, MAX_EQ_TOKEN_LENGTH

MODEL_PATH = './equation_parser/equation_parser_simple.h5'


class CaptionModelSimple:
    def create_model(self, input_shape=(100, 100, 3)):
        resnet_base = applications.resnet.ResNet50(
            include_top=False,
            input_shape=input_shape
        )

        vocab_size = len(TOKENS)

        self.model = models.Sequential([
            resnet_base,
            layers.Flatten(),
            # layers.Dense(1024, activation='relu'),
            # layers.Dropout(0.7),
            # layers.Dense(512, activation='relu'),
            # layers.Dropout(0.5),
            # layers.Dense(256, activation='relu'),
            # layers.Dropout(0.2),
            layers.Dense(MAX_EQ_TOKEN_LENGTH *
                         vocab_size, activation='softmax')
        ])

        for layer in resnet_base.layers:
            layer.trainable = False

        for layer in resnet_base.layers[-24:]:
            layer.trainable = True

        self.model.compile(optimizer='adam',
                           loss='mse',
                           metrics=['accuracy'])

    def load_model(self):
        if self.model_cached():
            print('Model cached. Loading model...')
            self.model = models.load_model(MODEL_PATH, compile=False)
            self.model.compile(optimizer='adam',
                               loss='mse',
                               metrics=['accuracy'])
        else:
            print('Model not cached. Creating and saving model...')
            self.create_model()

    def model_cached(self):
        return os.path.exists(MODEL_PATH)

    def save_model(self):
        self.model.save(MODEL_PATH)
