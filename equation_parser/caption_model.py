import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, layers, models, optimizers, applications
import string

from .tokens import MIN_EQ_TOKEN_LENGTH, TOKENS, VOCAB_SIZE, CONTEXT_WINDOW_LENGTH

MODEL_PATH = './equation_parser/equation_parser.h5'


class CaptionModel:
    def create_model(self):
        self.model = models.Sequential([
            layers.Input(
                shape=(MIN_EQ_TOKEN_LENGTH, 1 + 1024)),
            layers.LSTM(256),
            layers.Dense(VOCAB_SIZE, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

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
