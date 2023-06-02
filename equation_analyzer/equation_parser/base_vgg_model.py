from .constants import EQ_IMAGE_WIDTH, EQ_IMAGE_HEIGHT
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Concatenate, Conv2D, Input, Dropout, BatchNormalization, MaxPooling2D

MODEL_PATH = './equation_analyzer/equation_parser/vgg_base.h5'


class BaseVggModel:
    def create_model(self):
        # img_input = Input(shape=(EQ_IMAGE_WIDTH, EQ_IMAGE_HEIGHT, 3))
        # img_conc = Concatenate()([img_input, img_input, img_input])
        vgg_model = VGG16(input_shape=(EQ_IMAGE_WIDTH, EQ_IMAGE_HEIGHT, 3),
                          include_top=False)
        self.model = models.Model(
            inputs=vgg_model.inputs, outputs=vgg_model.layers[13].output)
        # self.model.build((None, EQ_IMAGE_WIDTH, EQ_IMAGE_HEIGHT, 1))
        self.model.summary()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            print('Conv model cached. Loading model...')
            self.model = models.load_model(MODEL_PATH, compile=False)
        else:
            print('Conv model not cached. Creating and saving model...')
            self.create_model()
            self.save_model()

    def save_model(self):
        self.model.save(MODEL_PATH)
