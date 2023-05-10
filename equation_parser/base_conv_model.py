import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, layers, models

MODEL_PATH = './equation_parser/conv_base.h5'


class BaseConvModel:
    def create_model(self):
        self.model = models.Sequential([
            layers.Conv2D(32, (7, 7), padding="same",
                          activation="relu", input_shape=(150, 150, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, (5, 5), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Flatten()
        ])

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