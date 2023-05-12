import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, layers, models, applications, optimizers
from tensorflow.keras.utils import plot_model

from .tokens import MAX_EQUATION_TEXT_LENGTH
from .base_resnet_model import BaseResnetModel
from .base_conv_model import BaseConvModel

MODEL_PATH = './equation_parser/caption_model.h5'
MODEL_IMG_PATH = './equation_parser/equation_parser.png'


class CaptionModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.base_resnet_model = BaseResnetModel()
        self.base_conv_model = BaseConvModel()

    def create_model(self):
        self.base_conv_model.load_model()

        N = 32
        HCF=234

        # extracted features
        inputs1 = layers.Input(shape=(300, 300, 3))
        resize = layers.Resizing(100,100)(inputs1)
        fe1 = self.base_conv_model.model(resize)
        fe2 = layers.Dense(N, activation='relu')(fe1)

        # LSTM
        inputs2 = layers.Input(batch_input_shape=(HCF, MAX_EQUATION_TEXT_LENGTH,))
        se1 = layers.Embedding(self.vocab_size, N,
                               input_length=MAX_EQUATION_TEXT_LENGTH, mask_zero=True)(inputs2)
        # se2 = layers.LSTM(512, return_sequences=True)(se1)
        se3 = layers.LSTM(N, stateful=True)(se1)
        se4 = layers.Dense(N, activation='relu')(se3)
        # se1 = layers.LSTM(512, return_sequences=True)(inputs2)
        # se2 = layers.LSTM(512, return_sequences=True)(se1)

        # merge
        decoder1 = layers.add([fe2, se4])
        decoder2 = layers.Dense(int(N), activation='relu')(decoder1)
        #  decoder3 = layers.Dropout(0.5)(decoder2)
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
