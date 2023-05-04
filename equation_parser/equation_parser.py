from matplotlib import pyplot as plt
import random
import math
import tensorflow as tf
import numpy as np

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

EQUATION_COUNT = 10000

EPOCHS = 1


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

        data_generator = DataGenerator(vocab_size)
        model = CaptionModel(vocab_size)

        print('Equation texts: train=', len(equation_texts))
        print('Photos: train=', len(equation_features))
        print('Vocabulary Size:', vocab_size)

        [a, b], c = next(data_generator.data_generator(
            equation_texts, equation_features, tokenizer))
        print(a.shape, b.shape, c.shape)

        model.load_model()
        train_generator = data_generator.data_generator(
            equation_texts, equation_features, tokenizer)
        validation_generator = data_generator.data_generator(
            equation_texts, equation_features, tokenizer)

        history = model.model.fit(
            train_generator, validation_data=validation_generator, epochs=EPOCHS, steps_per_epoch=steps, validation_steps=steps)
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
