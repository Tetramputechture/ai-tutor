import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers, applications


class ResnetModel:
    def create_model(self, input_shape=(224, 224, 3)):
        resnet_base = applications.resnet.ResNet50(
            include_top=False,
            input_shape=input_shape
        )
        model = models.Sequential([
            resnet_base,
            layers.Flatten(),
            layers.Dense(4, activation='relu')
        ])

        for layer in resnet_base.layers:
            layer.trainable = False

        for layer in resnet_base.layers[-24:]:
            layer.trainable = True

        # for layer in resnet_base.layers:
        #     if isinstance(layer, layers.BatchNormalization):
        #         layer.trainable = False

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])
        return model