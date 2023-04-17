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
from .caption_model_simple import CaptionModelSimple
from .base_resnet_model import BaseResnetModel
from .tokens import MAX_EQ_TOKEN_LENGTH, TOKENS

EQUATION_COUNT = 500

epochs = 10

batch_size = 64

test_size = 0.1

EQUATION_DATA_PATH = './equation_parser/data'


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def onehot_value_from_token(token):
    return TOKENS.index(token)


def tokens_from_onehot(onehot_tokens):
    result_tokens = []
    for onehot_token in onehot_tokens:
        print(onehot_token)
        index = list(onehot_token).index(1)
        result_tokens.append(TOKENS[index])
    result_tokens = filter(lambda t: t != 'PAD', result_tokens)
    return ''.join(result_tokens)


class EquationParserSimple:
    def train_model(self):
        self.eq_image_data = []
        self.eq_tokens = []
        self.next_tokens = []
        self.caption_model_simple = CaptionModelSimple()
        self.x = []
        self.y = []

        # Step 1: Fetch equation images
        print('Initializing equation image data...')

        # if self.data_cached():
        #     print('Cached equation sheet data found.')
        #     self.eq_image_data = np.load(EQUATION_IMAGE_DATA_PATH)
        #     self.eq_tokens = np.load(EQUATION_IMAGE_TOKENS_PATH)
        # else:

        generator = EquationImageGenerator()
        onehot_pad_value = TOKENS_ONEHOT[TOKENS.index('PAD')]

        if generator.images_cached():
            equations = generator.equations_from_cache()
            for equation in generator.equations_from_cache():
                (eq_image, eq_tokens) = equation
                eq_image = eq_image.resize(
                    (100, 100), resample=PIL.Image.BILINEAR)
                eq_image = eq_image.convert('RGB')
                eq_image_data = image.img_to_array(eq_image)
                self.eq_image_data.append(eq_image_data)

                full_tokens = [
                    onehot_pad_value for _ in range(MAX_EQ_TOKEN_LENGTH)]

                for idx, token in enumerate(list(eq_tokens)):
                    onehot_token_value = TOKENS_ONEHOT[TOKENS.index(token)]
                    full_tokens[idx] = onehot_token_value

                self.eq_tokens.append(np.array(full_tokens).flatten())
        else:
            for i in range(EQUATION_COUNT):
                eq_image, eq_tokens = generator.generate_equation_image()
                eq_image = eq_image.resize(
                    (100, 100), resample=PIL.Image.BILINEAR)
                eq_image = eq_image.convert('RGB')
                eq_image_data = image.img_to_array(eq_image)

                self.eq_image_data.append(eq_image_data)

                full_tokens = [
                    onehot_pad_value for _ in range(MAX_EQ_TOKEN_LENGTH)]

                for idx, token in enumerate(list(eq_tokens)):
                    onehot_token_value = TOKENS_ONEHOT[TOKENS.index(token)]
                    full_tokens[idx] = onehot_token_value

                self.eq_tokens.append(np.array(full_tokens).flatten())

        self.eq_image_data = np.array(
            self.eq_image_data).astype('float32')
        self.eq_tokens = np.array(
            self.eq_tokens).astype('float32')

        train_x, test_x, train_y, test_y = train_test_split(
            self.eq_image_data, self.eq_tokens, test_size=test_size
        )

        print(self.eq_tokens[0])
        print(self.eq_tokens[1])

        print(self.eq_image_data.shape)
        print(self.eq_tokens.shape)

        # # Step 3: Train model

        self.caption_model_simple.load_model()
        if not self.caption_model_simple.model_cached():
            print('Training model...')
            history = self.caption_model_simple.model.fit(train_x, train_y, epochs=epochs,
                                                          validation_data=(test_x, test_y), batch_size=batch_size)

            self.caption_model_simple.save_model()
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

            plt.show()

        eq_image, eq_tokens = generator.generate_equation_image()
        eq_image = eq_image.resize((100, 100), resample=PIL.Image.BILINEAR)
        eq_image_data = image.img_to_array(eq_image.convert('RGB'))
        predicted = self.infer_from_model(eq_image_data)
        plt.imshow(eq_image)
        plt.text(10, 10, f'Ground truth: {eq_tokens}')
        plt.text(10, 80, f'Predicted: {predicted}')
        plt.show()

        # test_loss, test_acc = self.caption_model.evaluate(
        #     test_image_data, test_eq_coords, verbose=2)

        # print(test_acc)

    def data_cached(self):
        return os.path.isdir(EQUATION_DATA_PATH)

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
        predictions = self.caption_model_simple.model.predict(imdata)[0]
        return tokens_from_onehot(np.split(predictions, MAX_EQ_TOKEN_LENGTH))
