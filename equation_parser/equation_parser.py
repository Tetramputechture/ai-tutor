from matplotlib import pyplot as plt
import random
import math
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import PIL
import os
import json
import sys
from datetime import datetime
import cv2
import itertools

import string

from .caption_model import CaptionModel
from .equation_preprocessor import EquationPreprocessor
from .equation_tokenizer import EquationTokenizer
from .ctc_data_generator import CtcDataGenerator
from .ctc_viz_callback import CtcVizCallback

from .constants import EQ_IMAGE_WIDTH, EQ_IMAGE_HEIGHT

TRAIN_CACHE_DIR = './equation_parser/data/images_train'
VAL_CACHE_DIR = './equation_parser/data/images_val'

TRAIN_EQUATION_COUNT = 350000
VAL_EQUATION_COUNT = 20000

BATCH_SIZE = 64

EPOCHS = 20


class EquationParser:
    def train_model(self):
        train_equation_preprocessor = EquationPreprocessor(
            TRAIN_EQUATION_COUNT, TRAIN_CACHE_DIR)
        train_equation_preprocessor.load_equations()
        train_equation_texts = train_equation_preprocessor.equation_texts

        val_equation_preprocessor = EquationPreprocessor(
            VAL_EQUATION_COUNT, VAL_CACHE_DIR)
        val_equation_preprocessor.load_equations()
        val_equation_texts = val_equation_preprocessor.equation_texts
        # equation_features = equation_preprocessor.equation_features

        tokenizer = EquationTokenizer(train_equation_texts).load_tokenizer()
        # + 2 for padding, ctc blank token
        vocab_size = len(tokenizer.word_index) + 2

        print('Vocab size: ', vocab_size)

        caption_model = CaptionModel(vocab_size, tokenizer)

        (model_input, model_output, model) = caption_model.create_model()

        # print(model.summary())

        train_data_generator = CtcDataGenerator(
            TRAIN_CACHE_DIR, train_equation_texts, tokenizer, BATCH_SIZE)
        val_data_generator = CtcDataGenerator(
            VAL_CACHE_DIR, val_equation_texts, tokenizer, BATCH_SIZE)

        train_num_batches = int(TRAIN_EQUATION_COUNT / BATCH_SIZE)
        val_num_batches = int(VAL_EQUATION_COUNT / BATCH_SIZE)

        viz_cb_train = CtcVizCallback(
            caption_model.test_func, train_data_generator.next_batch(), True, train_num_batches, tokenizer)
        viz_cb_val = CtcVizCallback(
            caption_model.test_func, val_data_generator.next_batch(), False, val_num_batches, tokenizer)

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

        model.fit(train_data_generator.next_batch(),
                  steps_per_epoch=train_num_batches,
                  epochs=EPOCHS,
                  callbacks=[viz_cb_train, viz_cb_val, train_data_generator,
                             val_data_generator, early_stop, model_chk_pt],
                  validation_data=val_data_generator.next_batch(),
                  validation_steps=val_num_batches)

        model.save('./equation_parser/caption_model.h5')

    def decode_label(self, tokenizer, model_output):
        out_best = list(np.argmax(model_output[0, 2:], axis=1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = tokenizer.sequences_to_texts([out_best])[0]
        outstr = outstr.replace(' ', '')
        outstr = outstr.replace('e', '')
        return outstr

    def test_model(self, model, img, label):
        start = datetime.now()
        accuracy = 0
        letter_acc = 0
        letter_cnt = 0
        count = 0
        img = cv2.imread(img)
        img_resized = cv2.resize(img, (EQ_IMAGE_WIDTH, EQ_IMAGE_HEIGHT))
        img = img_resized[:, :, 1]
        img = img.T
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        img = img / 255
        model_output = model.predict(img)
        tokenizer = EquationTokenizer().load_tokenizer()
        predicted_output = self.decode_label(tokenizer, model_output)
        actual_output = label
        letter_mismatch = 0
        for j in range(min(len(predicted_output), len(actual_output))):
            if predicted_output[j] == actual_output[j]:
                letter_acc += 1
            else:
                letter_mismatch += 1
        letter_cnt += max(len(predicted_output), len(actual_output))

        print('Actual: ', actual_output)
        print('Predicted: ', predicted_output)
        return predicted_output
