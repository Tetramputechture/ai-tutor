import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, layers, models, optimizers, applications, optimizers
from tensorflow.keras.utils import plot_model

from .tokens import MAX_EQUATION_TEXT_LENGTH
from .base_resnet_model import BaseResnetModel

MODEL_PATH = './equation_parser/caption_model.h5'
MODEL_IMG_PATH = './equation_parser/equation_parser.png'


class CaptionModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.base_resnet_model = BaseResnetModel()

    def create_model(self):
        # extracted features
        inputs1 = layers.Input(shape=(2048,))
        fe1 = layers.Dense(256, activation='relu')(inputs1)
        fe2 = layers.Dropout(0.5)(fe1)
        fe3 = layers.Dense(16, activation='relu')(fe2)

        # LSTM
        inputs2 = layers.Input(shape=(MAX_EQUATION_TEXT_LENGTH))
        se1 = layers.Embedding(self.vocab_size, 128,
                               input_length=MAX_EQUATION_TEXT_LENGTH)(inputs2)
        # se2 = layers.LSTM(64, return_sequences=True)(se1)
        se3 = layers.LSTM(16)(se1)
        # se1 = layers.LSTM(512, return_sequences=True)(inputs2)
        # se2 = layers.LSTM(512, return_sequences=True)(se1)

        # merge
        decoder1 = layers.add([fe3, se3])
        decoder2 = layers.Dense(16, activation='relu')(decoder1)
        outputs = layers.Dense(self.vocab_size, activation='softmax')(decoder2)

        self.model = models.Model(inputs=[inputs1, inputs2], outputs=outputs)

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        plot_model(self.model, to_file=MODEL_IMG_PATH, show_shapes=True)

    def load_model(self):
        if self.model_cached():
            print('Model cached. Loading model...')
            self.model = models.load_model(MODEL_PATH, compile=False)
            self.model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
        else:
            print('Model not cached. Creating model...')
            self.create_model()

    def model_cached(self):
        return os.path.exists(MODEL_PATH)

    def save_model(self):
        self.model.save(MODEL_PATH)
