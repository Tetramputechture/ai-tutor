import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers, applications


class ResnetModel:
    def create_model(self, input_shape=(224, 224, 3)):
        resnet_base = applications.resnet.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        chopped_resnet = models.Model(inputs=[resnet_base.input], outputs=[
                                      resnet_base.layers[90].output])

        model = models.Sequential([
            resnet_base,
            layers.Flatten(),
            # layers.Dense(256, activation='relu'),
            # layers.Dropout(0.2),
            layers.Dense(4, activation='relu')
        ])

        for layer in resnet_base.layers:
            layer.trainable = False

        for layer in resnet_base.layers[-48:]:
            layer.trainable = True

        # for layer in resnet_base.layers:
        #     if isinstance(layer, layers.BatchNormalization):
        #         layer.trainable = False

        optimizer = optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=['accuracy'])
        return model
