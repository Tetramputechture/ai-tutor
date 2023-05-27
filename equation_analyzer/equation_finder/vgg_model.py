import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers, applications


class VggModel:
    def create_model(self, input_shape=(224, 224, 3)):
        resnet_base = applications.VGG16(
            include_top=False,
            input_shape=input_shape
        )
        model = models.Sequential([
            resnet_base,
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(4, activation='relu')
        ])

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])
        return model
