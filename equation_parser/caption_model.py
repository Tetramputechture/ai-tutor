import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, layers, models, optimizers, applications
import string

from .tokens import MIN_EQ_TOKEN_LENGTH, TOKENS, VOCAB_SIZE, CONTEXT_WINDOW_LENGTH, MAX_EQ_TOKEN_LENGTH

MODEL_PATH = './equation_parser/equation_parser.h5'


class CaptionModel:
    def create_model(self):
        # extracted features from resnet
        inputs1 = layers.Input(shape=(256,))
        fe1 = layers.Dropout(0.5)(inputs1)
        fe2 = layers.Dense(16, activation='relu')(fe1)

        # LSTM
        inputs2 = layers.Input(shape=(MAX_EQ_TOKEN_LENGTH, 1))
        # se1 = layers.Embedding(VOCAB_SIZE, 256, mask_zero=True)(inputs2)
        se3 = layers.LSTM(16)(inputs2)

        # merge
        decoder1 = layers.add([fe2, se3])
        decoder2 = layers.Dense(32, activation='relu')(decoder1)
        outputs = layers.Dense(VOCAB_SIZE, activation='softmax')(decoder2)

        self.model = models.Model(inputs=[inputs1, inputs2], outputs=outputs)

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
