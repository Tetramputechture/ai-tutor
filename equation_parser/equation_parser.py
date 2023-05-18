from matplotlib import pyplot as plt
import random
import math
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

import PIL
import os
import json
import sys

import string

from .caption_model import CaptionModel
from .tokens import MAX_EQUATION_TEXT_LENGTH
from .equation_preprocessor import EquationPreprocessor
from .equation_tokenizer import EquationTokenizer
from .ctc_data_generator import CtcDataGenerator

TRAIN_CACHE_DIR = './equation_parser/data/images_train'
VAL_CACHE_DIR = './equation_parser/data/images_val'

TRAIN_EQUATION_COUNT = 5
VAL_EQUATION_COUNT = 5

EPOCHS = 5


class EquationParser:
    def train_model(self):
        train_equation_preprocessor = EquationPreprocessor(
            EQUATION_COUNT, TRAIN_CACHE_DIR)
        train_equation_preprocessor.load_equations()
        train_equation_texts = train_equation_preprocessor.equation_texts

        val_equation_preprocessor = EquationPreprocessor(
            EQUATION_COUNT, VAL_CACHE_DIR)
        val_equation_preprocessor.load_equations()
        val_equation_texts = train_equation_preprocessor.equation_texts
        # equation_features = equation_preprocessor.equation_features

        tokenizer = EquationTokenizer(equation_texts).load_tokenizer()
        vocab_size = len(tokenizer.word_index) + 1
        steps = len(equation_texts)

        print('Vocab size: ', vocab_size)

        train_data_generator = CtcDataGenerator(
            vocab_size, equation_texts, tokenizer)
        inputs, outputs = data_generator.full_dataset()

        early_stop = EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True)
        model_chk_pt = ModelCheckpoint(
            './equation_parser/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
            mode='auto',
            period=2)

        logdir = os.path.join(
            "./equation_parser/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        # print('Equation texts: train=', len(equation_texts))
        # print('Photos: train=', len(equation_features))
        # print('Vocabulary Size:', vocab_size)

        # next(data_generator.data_generator(
        #      equation_texts, equation_features, tokenizer))
        # print(a.shape, b.shape, c.shape)

        # train_generator = data_generator.data_generator(
        #     equation_texts, equation_features, tokenizer)
        # validation_generator = data_generator.data_generator(
        #     equation_texts, equation_features, tokenizer)

        # history = model.model.fit(
        #     train_generator, validation_data=validation_generator, epochs=EPOCHS, steps_per_epoch=steps, validation_steps=steps)

        def fit():
            history = model.model.fit(
                x=[train_x1, train_x2],
                y=train_y,
                validation_data=([test_x1, test_x2], test_y),
                epochs=1,
                shuffle=False,
                # steps_per_epoch=len(equation_texts),
                batch_size=BATCH_SIZE
            )
            return history

        for i in range(EPOCHS):
            history = fit()
            # model.model.reset_states()

        model.save_model()

        plt.subplot(2, 2, 1)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        plt.subplot(2, 2, 2)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.show()

        return train_x1, test_x2, train_y
