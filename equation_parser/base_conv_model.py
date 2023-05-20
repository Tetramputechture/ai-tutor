import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, layers, models

MODEL_PATH = './equation_parser/conv_base.h5'


class BaseConvModel:
    def create_model(self):
        # input is 100x100
        self.model = models.Sequential([
            layers.Resizing(84, 84),
            layers.Conv2D(56, (7, 7), padding="same",
                          activation="relu", input_shape=(84, 84, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, (5, 5), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(1, 1), padding='same'),

            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(1, 2)),

            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(1, 2)),
            # layers.Flatten()
        ])

        self.model.build((None, 100, 100, 3))
        self.model.summary()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            print('Conv model cached. Loading model...')
            self.model = models.load_model(MODEL_PATH, compile=False)
        else:
            print('Conv model not cached. Creating and saving model...')
            self.create_model()
            self.save_model()

    def save_model(self):
        self.model.save(MODEL_PATH)
