import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, models
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization, MaxPooling2D

SHEET_WIDTH = 224
SHEET_HEIGHT = 224


class ConvModel:
    def create_model(self):
        model = models.Sequential([
            Conv2D(64, (5, 5),
                   activation="relu", input_shape=(SHEET_WIDTH, SHEET_HEIGHT, 1)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation="relu"),
            # Dropout(DROPOUT_RATE),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation="relu"),
            # Dropout(DROPOUT_RATE),
            BatchNormalization(),
            MaxPooling2D(pool_size=(1, 2)),

            Conv2D(128, (3, 3), activation="relu"),
            # Dropout(DROPOUT_RATE),
            BatchNormalization(),
            MaxPooling2D(pool_size=(1, 2)),

            Flatten(),
            Dense(256, activation="relu"),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(4, activation='relu')
        ])

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])

        # model.summary()

        return model
