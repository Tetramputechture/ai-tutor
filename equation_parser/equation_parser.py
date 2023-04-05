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

from ..equation_finder.equation_image_generator import EquationImageGenerator

from .resnet_model import ResnetModel

equation_count = 500

epochs = 20

batch_size = 64

test_size = 0.3

EQUATION_IMAGE_DATA_PATH = './equation_parser/data/equation_image_data.npy'
EQUATION_IMAGE_TOKENS_PATH = './equation_parser/data/equation_tokens_path.npy'

TOKENS = list(string.digits + '+=/') + ['START'] + ['END']


class EquationParser:
    def __init__(self):
        self.model = ResnetModel().create_model()
        self.eq_image_data = []
        self.eq_tokens = []

    def train_model(self):
        # Step 1: Fetch equation images
        print('Initializing equation image data...')

        if self.data_cached():
            print('Cached equation sheet data found.')
            self.eq_image_data = np.load(EQUATION_IMAGE_DATA_PATH)
            self.eq_tokens = np.load(EQUATION_IMAGE_TOKENS_PATH)
        else:
            generator = EquationImageGenerator()
            for i in len(equation_count):
                eq_image, eq_tokens = generator.generate_equation_image()
                eq_image = eq_image.resize(
                    (100, 100), resample=PIL.Image.BILINEAR)
                eq_image = eq_image.convert('RGB')
                self.eq_image_data.append(image.img_to_array(eq_image))
                zeroes = np.zeroes(len(TOKENS))
                # each image y data is array of every token
                self.eq_tokens.append(eq_box.to_array())

            del sheets

            self.sheet_image_data = np.array(
                self.sheet_image_data).astype('float32')
            self.sheet_eq_coords = np.array(
                self.sheet_eq_coords).astype('float32')

            if not os.path.isdir(SHEET_DATA_PATH):
                os.makedirs(SHEET_DATA_PATH)

            np.save(SHEET_IMAGE_DATA_PATH, self.sheet_image_data)
            np.save(SHEET_EQ_COORDS_PATH, self.sheet_eq_coords)

        train_image_data, test_image_data, train_eq_coords, test_eq_coords = train_test_split(
            self.sheet_image_data, self.sheet_eq_coords, test_size=test_size
        )

    def data_cached(self):
        return os.path.isdir(SHEET_DATA_PATH) and \
            os.path.isfile(SHEET_IMAGE_DATA_PATH) and \
            os.path.isfile(SHEET_EQ_COORDS_PATH)

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            print('Model cached. Loading model...')
            self.model = models.load_model(MODEL_PATH, compile=False)
            self.model.compile()
        else:
            print('Model not cached. Training and saving model...')
            self.train_model()
            self.save_model()

    def save_model(self):
        self.model.save(MODEL_PATH)

    def infer_from_model(self, image_data):
        imdata = np.expand_dims(image_data, axis=0)
        imdata -= self.sheet_image_data.mean(
            axis=(0, -2, -1), keepdims=1)
        predictions = self.model.predict(imdata)[0]
        return EquationBox((int(predictions[0]), int(predictions[1])),
                           (int(predictions[2]), int(predictions[3])))
