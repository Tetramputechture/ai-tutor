import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, layers, models, optimizers, applications
from tensorflow.keras.utils import plot_model

from .tokens import MAX_EQUATION_TEXT_LENGTH

MODEL_PATH = './equation_parser/caption_model.h5'
MODEL_IMG_PATH = './equation_parser/equation_parser.png'


class CaptionModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def create_model(self):
        # extracted features from xception
        inputs1 = layers.Input(shape=(1000,))
        # fe1 = layers.Dropout(0.2)(inputs1)
        fe2 = layers.Dense(256, activation='relu')(inputs1)

        # LSTM
        inputs2 = layers.Input(shape=(MAX_EQUATION_TEXT_LENGTH, 1))
        # se1 = layers.Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
        # se2 = layers.Dropout(0.2)(inputs2)
        # se1 = layers.LSTM(512, return_sequences=True)(inputs2)
        # se2 = layers.LSTM(512, return_sequences=True)(se1)
        se3 = layers.LSTM(256)(inputs2)

        # merge
        decoder1 = layers.add([fe2, se3])
        decoder2 = layers.Dense(256, activation='relu')(decoder1)
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
