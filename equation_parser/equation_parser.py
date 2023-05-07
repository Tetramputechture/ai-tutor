from matplotlib import pyplot as plt
import random
import math
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

import PIL
import os
import json
import sys

import string

from .caption_model import CaptionModel
from .tokens import MAX_EQUATION_TEXT_LENGTH
from .equation_preprocessor import EquationPreprocessor
from .equation_tokenizer import EquationTokenizer
from .data_generator import DataGenerator

EQUATION_COUNT = 10

EPOCHS = 1

TEST_SIZE = 0.1


class EquationParser:
    def train_model(self):
        equation_preprocessor = EquationPreprocessor(
            EQUATION_COUNT)
        equation_preprocessor.load_equations()
        equation_texts = equation_preprocessor.equation_texts
        equation_features = equation_preprocessor.equation_features

        tokenizer = EquationTokenizer(equation_texts).load_tokenizer()
        vocab_size = len(tokenizer.word_index) + 1
        steps = len(equation_texts)

        data_generator = DataGenerator(
            vocab_size, equation_texts, equation_features, tokenizer)
        X1, X2, y = data_generator.full_dataset()
        model = CaptionModel(vocab_size)

        train_x1, test_x1, train_x2, test_x2, train_y, test_y = train_test_split(
            X1, X2, y, test_size=TEST_SIZE
        )

        # print('Equation texts: train=', len(equation_texts))
        # print('Photos: train=', len(equation_features))
        # print('Vocabulary Size:', vocab_size)

        # next(data_generator.data_generator(
        #      equation_texts, equation_features, tokenizer))
        # print(a.shape, b.shape, c.shape)

        model.load_model()
        # train_generator = data_generator.data_generator(
        #     equation_texts, equation_features, tokenizer)
        # validation_generator = data_generator.data_generator(
        #     equation_texts, equation_features, tokenizer)

        # history = model.model.fit(
        #     train_generator, validation_data=validation_generator, epochs=EPOCHS, steps_per_epoch=steps, validation_steps=steps)
        history = model.model.fit(
            x=[train_x1, train_x2], y=train_y, validation_data=([test_x1, test_x2], test_y), epochs=10)
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
