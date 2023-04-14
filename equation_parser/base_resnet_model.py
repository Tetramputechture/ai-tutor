import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, models, optimizers, applications

MODEL_PATH = './equation_parser/resnet_base.h5'


class BaseResnetModel:
    def create_model(self, input_shape=(100, 100, 3)):
        resnet_base = applications.resnet.ResNet50(
            include_top=False,
            input_shape=input_shape
        )

        self.model = models.Sequential([
            resnet_base,
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu')
        ])

        self.model.compile(optimizer='adam',
                           loss='mse',
                           metrics=['accuracy'])

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            print('Model cached. Loading model...')
            self.model = models.load_model(MODEL_PATH, compile=False)
            self.model.compile(optimizer='adam',
                               loss='mse',
                               metrics=['accuracy'])
        else:
            print('Model not cached. Creating and saving model...')
            self.create_model()

    def save_model(self):
        self.model.save(MODEL_PATH)
