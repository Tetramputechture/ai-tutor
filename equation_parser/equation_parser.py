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

EQUATION_COUNT = 100

EPOCHS = 50

TRAIN_SIZE = 0.7

BATCH_SIZE = 234


class EquationParser:

    def custom_train_test_split(self, x1, x2, y, train_size=0.8):
        x1_train, x1_test, x2_train, x2_test, y_train, y_test = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])

        # indices = np.arange(x1.shape[0])
        # np.random.shuffle(indices)

        # x1 = x1[indices]
        # x2 = x2[indices]
        # y = y[indices]

        split_ndx = int(len(y)*train_size)

        x1_train, x1_test, x2_train, x2_test, y_train, y_test = x1[:split_ndx],x1[split_ndx:],x2[:split_ndx],x2[split_ndx:],y[:split_ndx],y[split_ndx:]
        def computeHCF(x, y):
            '''
            Computes highest common factor...
            ref: https://datascience.stackexchange.com/questions/32831/batch-size-of-stateful-lstm-in-keras
            '''
            if x > y:
                smaller = y
            else:
                smaller = x
            for i in range(1, smaller+1):
                if((x % i == 0) and (y % i == 0)):
                    hcf = i

            print('\nHCF: ', hcf, '\n')
            return hcf

        batch_size= computeHCF(x1_train.shape[0], x1_test.shape[0])

        return x1_train, x1_test, x2_train, x2_test, y_train, y_test
    
    def train_model(self):
        equation_preprocessor = EquationPreprocessor(
            EQUATION_COUNT)
        equation_preprocessor.load_equations()
        equation_texts = equation_preprocessor.equation_texts
        # equation_features = equation_preprocessor.equation_features

        tokenizer = EquationTokenizer(equation_texts).load_tokenizer()
        vocab_size = len(tokenizer.word_index) + 1
        steps = len(equation_texts)

        data_generator = DataGenerator(
            vocab_size, equation_texts, tokenizer)
        X1, X2, y = data_generator.full_dataset()

        data_generator.save_data()
        model = CaptionModel(vocab_size)

        print('Shapes:')
        print(X1.shape)
        print(X2.shape)
        print(y.shape)

        train_x1, test_x1, train_x2, test_x2, train_y, test_y = self.custom_train_test_split(
            X1, X2, y, train_size=TRAIN_SIZE
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
            model.model.reset_states()


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
