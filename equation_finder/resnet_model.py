import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, applications


class ResnetModel:
    def create_model(self):
        resnet_base = applications.resnet.ResNet50(
            include_top=False,
            input_shape=(300, 300, 3)
        )
        model = models.Sequential([
            resnet_base,
            layers.Flatten(),
            # layers.Dropout(rate=0.7),
            layers.Dense(4, activation='relu')
        ])

        for layer in resnet_base.layers:
            layer.trainable = False

        for layer in resnet_base.layers[-24:]:
            layer.trainable = True

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])
        return model

    def create_model_v2(self):
        resnet_base = applications.resnet_v2.ResNet50V2()(
            include_top=False,
            input_shape=(300, 300, 3)
        )
        model = models.Sequential([
            resnet_base,
            layers.Flatten(),
            layers.Dropout(rate=0.2),
            layers.Dense(4, activation='relu')
        ])

        for layer in resnet_base.layers:
            layer.trainable = False

        for layer in resnet_base.layers[-24:]:
            layer.trainable = True

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])
        return model
