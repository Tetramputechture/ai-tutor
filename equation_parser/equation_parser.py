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
from .tokens import MAX_EQ_TOKEN_LENGTH, TOKENS, TOKENS_ONEHOT, MIN_EQ_TOKEN_LENGTH

EQUATION_COUNT = 100

STRIDE = 1

epochs = 10

batch_size = 64

test_size = 0.1


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def onehot_value_from_token(token):
    return TOKENS_ONEHOT[TOKENS.index(token)]


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
        self.generator = EquationImageGenerator()
        self.caption_model = CaptionModel()
        self.preprocess_x = []
        self.preprocess_y = []
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

        if not self.caption_model.model_cached():
            if self.generator.images_cached():
                equations = self.generator.equations_from_cache()
                for equation in self.generator.equations_from_cache()[:EQUATION_COUNT]:
                    self.append_equation_data_to_dataset(equation)
            else:
                for i in range(EQUATION_COUNT):
                    self.append_equation_data_to_dataset(
                        self.generator.generate_equation_image())

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
            plt.show()

        self.caption_model.load_model()

        eq_image, eq_tokens = self.generator.generate_equation_image()
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

        for idx, token in enumerate(list(eq_tokens)):
            onehot_token_value = TOKENS_ONEHOT[TOKENS.index(token)]
            concatenated_x_value = np.concatenate(
                (features_arr, onehot_token_value)
            )
            self.preprocess_x.append(concatenated_x_value)

            if idx < len(list(eq_tokens)) - 1:
                next_eq_token = list(eq_tokens)[idx + 1]
                next_eq_token_onehot_value = TOKENS_ONEHOT[TOKENS.index(
                    next_eq_token)]
                self.preprocess_y.append(next_eq_token_onehot_value)

        self.preprocess_y.append(TOKENS_ONEHOT[TOKENS.index('END')])

        for i in range(0, len(self.preprocess_x) - MIN_EQ_TOKEN_LENGTH, STRIDE):
            x_values = self.preprocess_x[i:i+MIN_EQ_TOKEN_LENGTH]
            self.x.append(x_values)
            self.y.append(self.preprocess_y[i+MIN_EQ_TOKEN_LENGTH])

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
        # populate starting context with random equation
        rand_eq_image, rand_eq_tokens = self.generator.generate_equation_image()
        rand_eq_image = rand_eq_image.resize(
            (100, 100), resample=PIL.Image.BILINEAR)
        rand_eq_image = rand_eq_image.convert('RGB')
        rand_eq_image_data = image.img_to_array(rand_eq_image)
        rand_features = self.base_resnet_model.model.predict(
            np.expand_dims(rand_eq_image_data, axis=0))
        rand_features_arr = np.array(rand_features[0]).astype('float32')

        preprocess_x = []
        for eq_token in rand_eq_tokens[len(rand_eq_tokens) - MIN_EQ_TOKEN_LENGTH:]:
            onehot_token_value = onehot_value_from_token(eq_token)
            preprocess_x.append(np.concatenate((
                rand_features_arr, onehot_token_value)))

        imdata = np.expand_dims(image_data, axis=0)
        features = self.base_resnet_model.model.predict(
            imdata)
        features_arr = np.array(features[0]).astype('float32')

        preprocess_x.append(np.concatenate((
            features_arr, TOKENS_ONEHOT[TOKENS.index('END')])))

        predictions = []

        for i in range(0, MAX_EQ_TOKEN_LENGTH, STRIDE):
            x_values = preprocess_x[i:i+MIN_EQ_TOKEN_LENGTH]
            prediction = self.caption_model.model.predict(
                np.expand_dims(np.array(x_values), axis=0))
            predicted_token = TOKENS[np.argmax(prediction)]
            onehot_pred_token = onehot_value_from_token(predicted_token)
            predictions.append(predicted_token)
            preprocess_x.append(np.concatenate(
                (features_arr, onehot_pred_token)
            ))

        print(np.array(predictions).shape)
        print(predictions)

        return ''.join(predictions)
