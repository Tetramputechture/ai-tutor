from matplotlib import pyplot as plt
import random
import math
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import PIL
import os
import json
import sys

import string

from .caption_model import CaptionModel
from .equation_preprocessor import EquationPreprocessor
from .equation_tokenizer import EquationTokenizer
from .ctc_data_generator import CtcDataGenerator
from .ctc_viz_callback import CtcVizCallback

TRAIN_CACHE_DIR = './equation_parser/data/images_train'
VAL_CACHE_DIR = './equation_parser/data/images_val'

TRAIN_EQUATION_COUNT = 100
VAL_EQUATION_COUNT = 100

BATCH_SIZE = 16

EPOCHS = 5


class EquationParser:
    def train_model(self):
        train_equation_preprocessor = EquationPreprocessor(
            TRAIN_EQUATION_COUNT, TRAIN_CACHE_DIR)
        train_equation_preprocessor.load_equations()
        train_equation_texts = train_equation_preprocessor.equation_texts

        val_equation_preprocessor = EquationPreprocessor(
            VAL_EQUATION_COUNT, VAL_CACHE_DIR)
        val_equation_preprocessor.load_equations()
        val_equation_texts = train_equation_preprocessor.equation_texts
        # equation_features = equation_preprocessor.equation_features

        tokenizer = EquationTokenizer(train_equation_texts).load_tokenizer()
        vocab_size = len(tokenizer.word_index) + 1

        print('Vocab size: ', vocab_size)

        caption_model = CaptionModel(vocab_size)

        (model_input, model_output, model) = caption_model.create_model()

        print(model.summary())

        train_data_generator = CtcDataGenerator(
            TRAIN_CACHE_DIR, train_equation_texts, tokenizer, BATCH_SIZE)
        val_data_generator = CtcDataGenerator(
            VAL_CACHE_DIR, val_equation_texts, tokenizer, BATCH_SIZE)

        train_num_batches = int(TRAIN_EQUATION_COUNT / BATCH_SIZE)
        val_num_batches = int(VAL_EQUATION_COUNT / BATCH_SIZE)

        viz_cb_train = CtcVizCallback(
            caption_model.test_func, train_data_generator.next_batch(), True, train_num_batches)
        viz_cb_val = CtcVizCallback(
            caption_model.test_func, val_data_generator.next_batch(), False, val_num_batches)

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

        logdir = os.path.join(
            "./equation_parser/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        model.fit(train_data_generator.next_batch(),
                  steps_per_epoch=train_num_batches,
                  epochs=EPOCHS,
                  callbacks=[viz_cb_train, viz_cb_val, train_data_generator,
                             val_data_generator, early_stop, model_chk_pt],
                  validation_data=val_data_generator,
                  validation_steps=val_num_batches)

        model.save('./equation_parser/best_model.h5')
