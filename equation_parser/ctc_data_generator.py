from .constants import MAX_EQUATION_TEXT_LENGTH
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

from PIL import Image
import random
import pandas as pd
import numpy as np

import cv2

from .constants import RNN_TIMESTEPS, EQ_IMAGE_HEIGHT, EQ_IMAGE_WIDTH

# http://man.hubwiz.com/docset/TensorFlow.docset/Contents/Resources/Documents/api_docs/python/tf/keras/backend/ctc_batch_cost.html


class CtcDataGenerator(keras.callbacks.Callback):
    def __init__(self, img_dir, equation_texts, tokenizer, batch_size):
        self.img_dir = img_dir
        self.equation_texts = [(k, v) for k, v in equation_texts.items()]
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.current_index = 0
        self.equation_count = len(equation_texts)
        self.indexes = list(range(self.equation_count))

    def fetch_and_preprocess_eq_image(self, eq_id):
        eq_image = cv2.imread(f'{self.img_dir}/{eq_id}.bmp')
        eq_image = eq_image[:, :, 1]  # Extracting Single Channel Image
        eq_image = cv2.resize(eq_image, (EQ_IMAGE_WIDTH, EQ_IMAGE_HEIGHT))
        eq_image = eq_image / 255
        return eq_image

    def next_data(self):
        self.current_index += 1
        # If current index becomes more than the number of images, make current index 0
        # and shuffle the indices list for random picking of image and text data
        if self.current_index >= self.equation_count:
            self.current_index = 0
            random.shuffle(self.indexes)
        return self.equation_texts[self.indexes[self.current_index]]

    def next_batch(self):
        while True:
            X_data = np.ones(
                [self.batch_size, EQ_IMAGE_WIDTH, EQ_IMAGE_HEIGHT, 1])
            Y_data = np.ones([self.batch_size, MAX_EQUATION_TEXT_LENGTH]) * -1

            # input_length for CTC which is the number of time-steps of the RNN output
            input_length = np.ones((self.batch_size, 1)) * (RNN_TIMESTEPS - 2)
            label_length = np.zeros((self.batch_size, 1))

            source_str = []

            for i in range(self.batch_size):
                eq_id, equation_text = self.next_data()

                # print(f'Fetching equation {eq_id} and adding to inputs...')

                # print('Equation text: ', equation_text)

                eq_image = self.fetch_and_preprocess_eq_image(eq_id)
                eq_image = eq_image.T
                eq_image = np.expand_dims(eq_image, -1)
                X_data[i] = eq_image

                # img_to_predict = img_to_predict / 127.5
                # img_to_predict = img_to_predict - 1.0
                # encode the sequence
                sequence = self.tokenizer.texts_to_sequences([equation_text])[
                    0]
                lbl_len = len(sequence)

                # print('Equation length: ', lbl_len)

                Y_data[i, 0:lbl_len] = sequence
                label_length[i] = lbl_len
                source_str.append(equation_text)

            inputs = {
                'img_input': X_data,
                'ground_truth_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                'source_str': source_str  # used for viz only
            }
            # prepare output for the Model and initialize to zeros
            outputs = {'ctc': np.zeros([self.batch_size])}

            yield (inputs, outputs)

    def full_dataset(self):
        X1, y, input_length, label_length, source_str = list(), list(), list(), list(), list()

        input_length = np.ones((self.batch_size, 1)) * 40
        label_length = np.zeros((self.batch_size, 1))

        for eq_id, equation_text in self.equation_texts.items():
            eq_image = fetch_and_preprocess_eq_image(eq_id)
            X1.append(eq_image)
            # img_to_predict = img_to_predict / 127.5
            # img_to_predict = img_to_predict - 1.0
            # encode the sequence
            sequence = self.tokenizer.texts_to_sequences([equation_text])[0]
            lbl_len = len(sequence)

            y.append(sequence)
            label_length.append(lbl_len)
            source_str.append(equation_text)

        inputs = {
            'img_input': X1,
            'ground_truth_labels': y,
            'input_length': input_length,
            'label_length': label_length,
            'source_str': source_str  # used for viz only
        }
        outputs = {'ctc': np.zeros([self.batch_size])}

        return (inputs, outputs)
