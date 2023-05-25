from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import random
import math
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

import PIL
import os
import json
import sys

from .equation_box import EquationBox
from .equation_sheet_generator import EquationSheetGenerator


from .resnet_model import ResnetModel
from .conv_model import ConvModel
from .vgg_model import VggModel

sheet_count = 5000

epochs = 20

BATCH_SIZE = 64

STEPS_PER_EPOCH = int(sheet_count / BATCH_SIZE)

test_size = 0.1

MODEL_PATH = './equation_analyzer/equation_finder/equation_finder.h5'
SHEET_DATA_PATH = './equation_analyzer/equation_finder/data'
SHEET_IMAGE_DATA_PATH = f'{SHEET_DATA_PATH}/sheet_image_data.npy'
SHEET_EQ_COORDS_PATH = f'{SHEET_DATA_PATH}/sheet_eq_coords.npy'


def coord_diff(coord, inferred_coords):
    return {
        'x1': abs(coord['x1'] - inferred_coords['x1']),
        'y1': abs(coord['y1'] - inferred_coords['y1']),
        'x2': abs(coord['x2'] - inferred_coords['x2']),
        'y2': abs(coord['y2'] - inferred_coords['y2'])
    }


def coord_accuracy(coord, coord_diff):
    return {
        'x1': (coord_diff['x1'] / coord['x1']) * 100,
        'y1': (coord_diff['y1'] / coord['y1']) * 100,
        'x2': (coord_diff['x2'] / coord['x2']) * 100,
        'y2': (coord_diff['y2'] / coord['y2']) * 100
    }


class EquationFinder:
    def __init__(self):
        self.model = ConvModel().create_model()
        self.sheet_image_data = []
        self.sheet_eq_coords = []

    def train_model(self):
        # Step 1: Fetch equation sheets
        print('Initializing equation sheet image data...')

        if self.data_cached():
            print('Cached equation sheet data found.')
            # self.sheet_image_data = np.load(SHEET_IMAGE_DATA_PATH)
            # self.sheet_eq_coords = np.load(SHEET_EQ_COORDS_PATH)
        else:
            sheet_images_path = './equation_analyzer/equation_finder/data/equation-sheet-images'
            sheets = EquationSheetGenerator(
                sheet_size=(224, 224),
                cache_dir=sheet_images_path
            ).generate_sheets(sheet_count)

            for sheet in sheets:
                sheet_image, eq_box = sheet
                sheet_image = sheet_image.convert('L')
                sheet_image = image.img_to_array(sheet_image)
                # Extracting Single Channel Image
                # sheet_image = sheet_image[:, :, 1]
                sheet_image = sheet_image / 255
                self.sheet_image_data.append(sheet_image)
                self.sheet_eq_coords.append(eq_box.to_array())

            if not os.path.isdir(SHEET_DATA_PATH):
                os.makedirs(SHEET_DATA_PATH)

            # np.save(SHEET_IMAGE_DATA_PATH, self.sheet_image_data)
            # np.save(SHEET_EQ_COORDS_PATH, self.sheet_eq_coords)

        # Step 2: Prepare train and test data

        train_image_data, test_image_data, train_eq_coords, test_eq_coords = train_test_split(
            self.sheet_image_data, self.sheet_eq_coords, test_size=test_size
        )

        # Step 3: Train model

        train_gen = self.data_generator(train_image_data, train_eq_coords)
        val_gen = self.data_generator(test_image_data, test_eq_coords)

        callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=2, restore_best_weights=True)

        history = self.model.fit(train_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=epochs, validation_steps=STEPS_PER_EPOCH,
                                 validation_data=val_gen, batch_size=BATCH_SIZE, callbacks=[callback])

        plt.subplot(2, 2, 1)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        plt.subplot(2, 2, 2)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        test_loss, test_acc = self.model.evaluate(
            val_gen, batch_size=BATCH_SIZE, steps=STEPS_PER_EPOCH, verbose=2)

        print(test_acc)

    def data_generator(self, image_data, coords):
        while True:
            X = []
            Y = []
            start = 0
            end = BATCH_SIZE

            while start < len(image_data):
                X = np.array(image_data[start:end])
                Y = np.array(coords[start:end])

                yield (X, Y)

                start += BATCH_SIZE
                end += BATCH_SIZE

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

    def infer_from_model(self, image_data) -> EquationBox:
        imdata = image.img_to_array(image_data)
        imdata = imdata[:, :, 1]
        imdata = imdata / 255
        imdata = np.expand_dims(imdata, axis=0)
        predictions = self.model.predict(imdata)[0]
        return EquationBox((int(predictions[0]), int(predictions[1])),
                           (int(predictions[2]), int(predictions[3])))

    def show_validation(self):
        validation_sheet = EquationSheetGenerator().clean_sheet_with_equation()

        rand_test_image = validation_sheet[0]
        rand_test_coords = validation_sheet[1].to_eq_coord()
        rand_test_image_data = image.img_to_array(
            rand_test_image.convert('RGB'))

        fig, ax = plt.subplots()
        ax.imshow(rand_test_image)

        xy = (rand_test_coords['x1'], rand_test_coords['y1'])
        width = rand_test_coords['x2'] - xy[0]
        height = rand_test_coords['y2'] - xy[1]
        ax.add_patch(Rectangle(xy, width, height,
                               fill=False, edgecolor="r"))

        fig, ax = plt.subplots()
        ax.imshow(rand_test_image)

        inferred_coords = self.infer_from_model(
            rand_test_image_data).to_eq_coord()

        xy = (inferred_coords['x1'], inferred_coords['y1'])
        width = inferred_coords['x2'] - xy[0]
        height = inferred_coords['y2'] - xy[1]
        ax.add_patch(Rectangle(xy, width, height,
                               fill=False, edgecolor="r"))

        print('Ground truth:')
        print(rand_test_coords)
        print('Inferred:')
        print(inferred_coords)
        print('Differential:')

        differential = [coord_diff(coord, inferred_coord)
                        for coord, inferred_coord in zip([rand_test_coords], [inferred_coords])]
        print(list(differential))

        accuracies = [coord_accuracy(coord, inferred_coord)
                      for coord, inferred_coord in zip([rand_test_coords], [inferred_coords])]
        print('Accuracy (%):')
        print(accuracies)

        plt.show()
