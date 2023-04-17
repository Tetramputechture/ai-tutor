import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, layers, models, optimizers, applications
import string

from .tokens import MAX_EQ_TOKEN_LENGTH, TOKENS, VOCAB_SIZE

MODEL_PATH = './equation_parser/equation_parser.h5'


class CaptionModel:
    def create_model(self):
        self.model = models.Sequential([
            layers.Input(
                shape=(MAX_EQ_TOKEN_LENGTH * VOCAB_SIZE + 256, 1)),
            # layers.Embedding(VOCAB_SIZE, 256, mask_zero=True),
            layers.LSTM(256),
            layers.Dense(VOCAB_SIZE, activation='softmax')
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
