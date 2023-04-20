from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import random
import math
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers, applications
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import PIL
import os
import json
import sys

import string

from .equation_image_generator import EquationImageGenerator
from .caption_model import CaptionModel
from .base_resnet_model import BaseResnetModel
from .tokens import MAX_EQ_TOKEN_LENGTH, TOKENS, TOKENS_ONEHOT, MIN_EQ_TOKEN_LENGTH

TRAIN = False

if "train" in str(sys.argv[1]).lower():
    TRAIN = True

EQUATION_COUNT = 100
STRIDE = 1

epochs = 15

batch_size = 32

test_size = 0.1


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def onehot_value_from_token(token):
    return TOKENS_ONEHOT[TOKENS.index(token)]


def tokens_from_onehot(onehot_tokens):
    result_tokens = []
    for onehot_token in onehot_tokens:
        result_tokens.append(TOKENS[np.argmax(onehot_token)])
    result_tokens = filter(lambda t: t != 'PAD', result_tokens)
    return ''.join(result_tokens)


class EquationParser:
    def train_model(self):
        self.next_tokens = []
        self.base_resnet_model = BaseResnetModel()
        self.generator = EquationImageGenerator()
        self.caption_model = CaptionModel()
        self.all_eq_tokens = []
        self.all_eq_features = []
        self.preprocess_x_tokens = []
        self.preprocess_x = []
        self.preprocess_y = []
        self.x = []
        self.y = []
        self.onehot_pad_value = TOKENS_ONEHOT[TOKENS.index('PAD')]
        self.vectorizer = TfidfVectorizer()

        # Step 1: Fetch equation images
        print('Initializing equation image data...')

        # if self.data_cached():
        #     print('Cached equation sheet data found.')
        #     self.eq_image_data = np.load(EQUATION_IMAGE_DATA_PATH)
        #     self.eq_tokens = np.load(EQUATION_IMAGE_TOKENS_PATH)
        # else:

        self.base_resnet_model.load_model()

        if TRAIN and not self.caption_model.model_cached():
            equations = []
            if self.generator.images_cached():
                equations = self.generator.equations_from_cache()
                for equation in self.generator.equations_from_cache()[:EQUATION_COUNT]:
                    self.append_equation_data_to_dataset(equation)
            else:
                for i in range(EQUATION_COUNT):
                    self.append_equation_data_to_dataset(
                        self.generator.generate_equation_image())

            self.vectorizer.fit_transform(
                self.preprocess_x_tokens)
            vectorized_start_token = np.asarray(self.vectorizer.transform(
                ['START']).todense())[0]
            onehot_end_value = TOKENS_ONEHOT[TOKENS.index('END')]

            for idx, equation in enumerate(self.generator.equations_from_cache()[:EQUATION_COUNT]):
                eq_features_arr = self.all_eq_features[idx]
                eq_tokens = equation[1]

                self.preprocess_x.append(np.concatenate(
                    (eq_features_arr, vectorized_start_token)))
                self.preprocess_y.append(
                    TOKENS_ONEHOT[TOKENS.index(eq_tokens[0])])

                for idx, eq_token in enumerate(eq_tokens):
                    vectorized_token = np.asarray(self.vectorizer.transform(
                        [eq_token]).todense())[0]
                    self.preprocess_x.append(np.concatenate(
                        (eq_features_arr, vectorized_token)))
                    if idx < len(eq_tokens) - 1:
                        onehot_next_value = TOKENS_ONEHOT[TOKENS.index(
                            eq_tokens[idx + 1])]
                        self.preprocess_y.append(onehot_next_value)

                self.preprocess_y.append(onehot_end_value)

            for i in range(0, len(self.preprocess_x) - MIN_EQ_TOKEN_LENGTH, STRIDE):
                x_values = self.preprocess_x[i:i+MIN_EQ_TOKEN_LENGTH]
                # if self.preprocess_x_tokens[i] == 'START':
                #     end_idx = self.preprocess_x_tokens[i:].index('END')
                #     current_equation = self.preprocess_x_tokens[i:end_idx]
                # print('Current equation:', current_equation)
                # print('Current x_values:', ''.join(
                #     self.preprocess_x_tokens[i:i+MIN_EQ_TOKEN_LENGTH]))
                # print('Next preprocess_x_tokens:', ''.join(
                #     self.preprocess_x_tokens[i+MIN_EQ_TOKEN_LENGTH+1:i+MIN_EQ_TOKEN_LENGTH+6]))
                # print('Next y tokens', ''.join(list(map(lambda y: TOKENS[np.argmax(
                #     y)], self.preprocess_y[i+MIN_EQ_TOKEN_LENGTH:i+MIN_EQ_TOKEN_LENGTH+4]))))
                # exit()
                self.x.append(x_values)
                self.y.append(self.preprocess_y[i+MIN_EQ_TOKEN_LENGTH])

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
        self.all_eq_features.append(features_arr)

        # preprocess_x = []
        # preprocess_y = []

        onehot_token_value = TOKENS_ONEHOT[TOKENS.index('START')]
        concatenated_x_value = np.concatenate(
            (features_arr, onehot_token_value)
        )
        # self.preprocess_x.append(concatenated_x_value)
        self.preprocess_x_tokens.append('START')
        # self.preprocess_y.append(TOKENS_ONEHOT[TOKENS.index(eq_tokens[0])])

        eq_tokens_to_append = []
        for idx, token in enumerate(list(eq_tokens)):
            onehot_token_value = TOKENS_ONEHOT[TOKENS.index(token)]
            concatenated_x_value = np.concatenate(
                (features_arr, onehot_token_value)
            )
            # self.preprocess_x.append(concatenated_x_value)
            self.preprocess_x_tokens.append(token)
            eq_tokens_to_append.append(token)
            if idx < len(list(eq_tokens)) - 1:
                next_eq_token = list(eq_tokens)[idx + 1]
                next_eq_token_onehot_value = TOKENS_ONEHOT[TOKENS.index(
                    next_eq_token)]
                # self.preprocess_y.append(next_eq_token_onehot_value)

        # self.preprocess_y.append(TOKENS_ONEHOT[TOKENS.index('END')])
        # self.all_eq_tokens.append(eq_tokens_to_append)
        # print('Original equation:', eq_tokens)
        # for i in range(0, len(preprocess_x) - MIN_EQ_TOKEN_LENGTH, STRIDE):
        #     x_values = preprocess_x[i:i+MIN_EQ_TOKEN_LENGTH]
        #     # print('Current x_values:', eq_tokens[i:i+MIN_EQ_TOKEN_LENGTH])
        #     # print('Next token', TOKENS[np.argmax(
        #     #     preprocess_y[i+MIN_EQ_TOKEN_LENGTH])])
        #     self.x.append(x_values)
        #     self.y.append(preprocess_y[i+MIN_EQ_TOKEN_LENGTH])

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
        for eq_token in rand_eq_tokens[:MIN_EQ_TOKEN_LENGTH]:
            onehot_token_value = onehot_value_from_token(eq_token)
            preprocess_x.append(np.concatenate((
                rand_features_arr, onehot_token_value)))

        preprocess_x.append(np.concatenate((
            rand_features_arr, TOKENS_ONEHOT[TOKENS.index('END')])))

        imdata = np.expand_dims(image_data, axis=0)
        features = self.base_resnet_model.model.predict(
            imdata)
        features_arr = np.array(features[0]).astype('float32')

        preprocess_x.append(np.concatenate((
            features_arr, TOKENS_ONEHOT[TOKENS.index('START')])))

        predictions = []

        for i in range(0, MAX_EQ_TOKEN_LENGTH, STRIDE):
            x_values = preprocess_x[i:i+MIN_EQ_TOKEN_LENGTH]
            prediction = self.caption_model.model.predict(
                np.expand_dims(np.array(x_values), axis=0))
            predicted_token = TOKENS[np.argmax(prediction)]
            if predicted_token == 'END':
                break

            onehot_pred_token = onehot_value_from_token(predicted_token)
            predictions.append(predicted_token)
            preprocess_x.append(np.concatenate(
                (features_arr, onehot_pred_token)
            ))

        print(np.array(predictions).shape)
        print(''.join(predictions))

        return ''.join(predictions)
