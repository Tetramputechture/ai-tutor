import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, layers, models, optimizers, applications
import string

from .tokens import MAX_EQ_TOKEN_LENGTH_PLUS_PAD, TOKENS

MODEL_PATH = './equation_parser/equation_parser.h5'


class CaptionModel:
    def create_model(self):
        vocab_size = len(TOKENS)

        self.model = models.Sequential([
            layers.Input(
                shape=(MAX_EQ_TOKEN_LENGTH_PLUS_PAD * vocab_size + 256, 1)),
            layers.LSTM(256),
            layers.Dense(vocab_size, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            print('Model cached. Loading model...')
            self.model = models.load_model(MODEL_PATH, compile=False)
            self.model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
        else:
            print('Model not cached. Creating model...')
            self.create_model()

    def save_model(self):
        self.model.save(MODEL_PATH)
