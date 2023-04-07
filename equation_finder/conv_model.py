import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models


class ConvModel:
    def create_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (7, 7), padding="same",
                          activation="relu", input_shape=(224, 224, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, (5, 5), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Flatten(),

            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),

            layers.Dense(4, activation='relu')
        ])

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])

        return model
