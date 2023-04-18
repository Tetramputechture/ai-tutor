from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import random
import math
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers, applications
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

import PIL
import os
import json
import sys

import string

from .equation_image_generator import EquationImageGenerator
from .caption_model import CaptionModel
from .base_resnet_model import BaseResnetModel
from .tokens import MAX_EQ_TOKEN_LENGTH, TOKENS, TOKENS_ONEHOT

EQUATION_COUNT = 1000

CONTEXT_WINDOW_LENGTH = 5

epochs = 50

batch_size = 64

test_size = 0.1


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def onehot_value_from_token(token):
    return TOKENS.index(token)


def tokens_from_onehot(onehot_tokens):
    result_tokens = []
    for onehot_token in onehot_tokens:
        index = find_nearest(onehot_token, 1)
        result_tokens.append(TOKENS[index])
    result_tokens = filter(lambda t: t != 'PAD', result_tokens)
    return ''.join(result_tokens)


class EquationParser:
    def train_model(self):
        self.next_tokens = []
        self.base_resnet_model = BaseResnetModel()
        self.caption_model = CaptionModel()
        self.x = []
        self.y = []
        self.onehot_pad_value = TOKENS_ONEHOT[TOKENS.index('PAD')]

        # Step 1: Fetch equation images
        print('Initializing equation image data...')

        # if self.data_cached():
        #     print('Cached equation sheet data found.')
        #     self.eq_image_data = np.load(EQUATION_IMAGE_DATA_PATH)
        #     self.eq_tokens = np.load(EQUATION_IMAGE_TOKENS_PATH)
        # else:

        self.base_resnet_model.load_model()
        generator = EquationImageGenerator()
        if generator.images_cached():
            equations = generator.equations_from_cache()
            for equation in generator.equations_from_cache()[:EQUATION_COUNT]:
                self.append_equation_data_to_dataset(equation)
        else:
            for i in range(EQUATION_COUNT):
                self.append_equation_data_to_dataset(
                    generator.generate_equation_image())

        # self.x = np.array(self.x).flatten()
        self.x = np.array(self.x)
        self.y = np.array(self.y).astype('float32')

        print('Shapes:')
        # total samples = equation_count * vocab_size
        # x: [total samples, ctx_window_length,]
        # y: [total samples, vocab_size]
        print(self.x.shape)
        print(self.y.shape)

        train_x, test_x, train_y, test_y = train_test_split(
            self.x, self.y, test_size=test_size
        )

        # # Step 3: Train model

        self.caption_model.load_model()
        history = self.caption_model.model.fit(train_x, train_y, epochs=epochs,
                                               validation_data=(test_x, test_y), batch_size=batch_size)

        self.caption_model.save_model()
        plt.subplot(2, 4, 1)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        plt.subplot(2, 4, 2)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        # eq_image, eq_tokens = generator.generate_equation_image()
        # eq_image = eq_image.resize((100, 100), resample=PIL.Image.BILINEAR)
        # eq_image_data = image.img_to_array(eq_image.convert('RGB'))
        # predicted = self.infer_from_model(eq_image_data)
        # plt.subplot(2, 4, 3)
        # plt.imshow(eq_image)
        # plt.text(10, 10, f'Ground truth: {eq_tokens}')
        # plt.text(10, 80, f'Predicted: {predicted}')
        plt.show()

        # test_loss, test_acc = self.caption_model.evaluate(
        #     test_image_data, test_eq_coords, verbose=2)

        # print(test_acc)

    def append_equation_data_to_dataset(self, equation):
        (eq_image, eq_tokens) = equation
        eq_image = eq_image.resize(
            (100, 100), resample=PIL.Image.BILINEAR)
        eq_image = eq_image.convert('RGB')
        eq_image_data = image.img_to_array(eq_image)
        image_to_predict = np.expand_dims(eq_image_data, axis=0)

        features = self.base_resnet_model.model.predict(
            image_to_predict)
        features_arr = np.array(features[0]).astype('float32')

        full_tokens = [
            self.onehot_pad_value for _ in range(MAX_EQ_TOKEN_LENGTH)]
        full_tokens[0] = TOKENS_ONEHOT[TOKENS.index('START')]
        x_window = []
        for ft in full_tokens:
            x_value = np.concatenate(
                (features_arr, np.array(full_tokens).flatten().astype('float32')))
            x_window.append(x_value)

        self.x.append(x_window)

        first_eq_token = list(eq_tokens)[0]
        first_eq_token_onehot_value = TOKENS_ONEHOT[TOKENS.index(
            first_eq_token)]
        self.y.append(first_eq_token_onehot_value)

        for idx, token in enumerate(list(eq_tokens)):
            onehot_token_value = TOKENS_ONEHOT[TOKENS.index(token)]
            full_tokens[idx + 1] = onehot_token_value

            x_window = []
            for ft in full_tokens:
                x_value = np.concatenate(
                    (features_arr, np.array(full_tokens).flatten().astype('float32')))
                x_window.append(x_value)

            self.x.append(x_window)

            if idx < len(list(eq_tokens)) - 1:
                next_eq_token = list(eq_tokens)[idx + 1]
                next_eq_token_onehot_value = TOKENS_ONEHOT[TOKENS.index(
                    next_eq_token)]
                self.y.append(next_eq_token_onehot_value)

        self.y.append(TOKENS_ONEHOT[TOKENS.index('END')])

    def data_cached(self):
        return os.path.isdir(SHEET_DATA_PATH) and \
            os.path.isfile(SHEET_IMAGE_DATA_PATH) and \
            os.path.isfile(SHEET_EQ_COORDS_PATH)

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            print('Model cached. Loading model...')
            self.caption_model = models.load_model(MODEL_PATH, compile=False)
            self.caption_model.compile()
        else:
            print('Model not cached. Training and saving model...')
            self.train_model()
            self.save_model()

    def save_model(self):
        self.caption_model.save(MODEL_PATH)

    def infer_from_model(self, image_data):
        imdata = np.expand_dims(image_data, axis=0)
        predictions = self.caption_model.model.predict(imdata)[0]
        return tokens_from_onehot(np.split(predictions, MAX_EQ_TOKEN_LENGTH))
