from .constants import EQ_IMAGE_WIDTH, EQ_IMAGE_HEIGHT
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, models
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, MaxPooling2D

MODEL_PATH = './equation_analyzer/equation_parser/conv_base.h5'

DROPOUT_RATE = 0.25


class BaseConvModel:
    def create_model(self):
        self.model = models.Sequential([
            Conv2D(64, (3, 3), padding="same",
                   activation="relu", input_shape=(EQ_IMAGE_WIDTH, EQ_IMAGE_HEIGHT, 1)),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding="same",
                   activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(256, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            Conv2D(256, (3, 3), padding="same", activation="relu"),
            Dropout(DROPOUT_RATE),
            BatchNormalization(),
            MaxPooling2D(pool_size=(1, 2)),

            Conv2D(512, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            Conv2D(512, (3, 3), padding="same", activation="relu"),
            Dropout(DROPOUT_RATE),
            BatchNormalization(),
            MaxPooling2D(pool_size=(1, 2)),

            Conv2D(512, (2, 2), padding='same', activation='relu'),
            Dropout(0.25),
            BatchNormalization()
        ])

        self.model.build((None, EQ_IMAGE_WIDTH, EQ_IMAGE_HEIGHT, 1))
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
