from .tokens import MAX_EQUATION_TEXT_LENGTH
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

from PIL import Image
import random
import pandas as pd
import numpy as np

# http://man.hubwiz.com/docset/TensorFlow.docset/Contents/Resources/Documents/api_docs/python/tf/keras/backend/ctc_batch_cost.html


def fetch_and_preprocess_eq_image(eq_id):
    eq_image = Image.open(f'./equation_parser/data/images/{eq_id}.bmp')
    eq_image = np.array(eq_image.resize((100, 100)))
    return eq_image


class CtcDataGenerator(keras.callbacks.Callback):
    def __init__(self, equation_dir, equation_count, tokenizer, batch_size):
        self.equation_dir = equation_dir
        self.equation_count = equation_count
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.indexes = list(range(self.equation_count))

    def load_data(self):
        equation_preprocessor = EquationPreprocessor(
            self.equation_count, self.equation_dir)
        equation_preprocessor.load_equations()
        self.equation_texts = equation_preprocessor.equation_texts

    def next_batch(self):
        while True:
            X1, y, input_length, label_length, source_str = list(), list(), list(), list(), list()

            input_length = np.ones((self.batch_size, 1)) * 40
            label_length = np.zeros((self.batch_size, 1))

            for eq_id, equation_text in self.equation_texts.items()[:self.batch_size]:
                eq_image = fetch_and_preprocess_eq_image(eq_id)
                X1.append(eq_image)
                # img_to_predict = img_to_predict / 127.5
                # img_to_predict = img_to_predict - 1.0
                # encode the sequence
                sequence = self.tokenizer.texts_to_sequences([equation_text])[
                    0]
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
