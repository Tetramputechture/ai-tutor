import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, models, optimizers, applications

MODEL_PATH = './equation_parser/resnet_base.h5'


class BaseResnetModel:
    def create_model(self, input_shape=(150, 150, 3)):
        resnet_base = applications.resnet.ResNet50(
            include_top=False,
            input_shape=input_shape,
            pooling='max',
        )

        self.model = models.Sequential([
            resnet_base,
            layers.Flatten(),
        ])

        self.model.compile(optimizer='adam',
                           loss='mse',
                           metrics=['accuracy'])

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            print('Resnet model cached. Loading model...')
            self.model = models.load_model(MODEL_PATH, compile=False)
            self.model.compile(optimizer='adam',
                               loss='mse',
                               metrics=['accuracy'])
        else:
            print('Resnet model not cached. Creating and saving model...')
            self.create_model()
            self.save_model()

    def save_model(self):
        self.model.save(MODEL_PATH)

