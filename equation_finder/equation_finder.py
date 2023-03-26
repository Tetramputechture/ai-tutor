from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import random
import math
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image


import PIL
import os
import json
import sys

from .bounding_rect import BoundingRect
from .equation_image_generator import EquationImageGenerator
from .equation_sheet_generator import EquationSheetGenerator


from .resnet_model import ResnetModel
from .conv_model import ConvModel

max_equations_per_sheet = 1
sheet_count = 5

epochs = 1

train_split = 0.7


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
        self.model = ResnetModel().create_model()

    def train_model(self):
        # Step 1: Fetch equation sheets
        print('Initializing equation sheet image data...')

        sheet_images_path = './data/equation-sheet-images'
        sheets = EquationSheetGenerator(
            max_equations_per_sheet,
            sheet_size=(200, 200),
            cache_dir=sheet_images_path
        ).generate_sheets(sheet_count)

        # Step 2: Prepare train and test data

        sheet_image_data = []
        sheet_eq_coords = []

        for sheet in sheets:
            eq_image, eq_coords = sheet
            sample = dict()
            eq_image = eq_image.convert('RGB')
            im_arr = image.img_to_array(eq_image)
            sheet_image_data.append(im_arr)
            coords = []
            for coord in eq_coords:
                coords.extend(
                    [coord['x1'], coord['y1'], coord['x2'], coord['y2']])
            sheet_eq_coords.append(coords)

        sheet_image_data = np.array(sheet_image_data).astype('float32')
        sheet_eq_coords = np.array(sheet_eq_coords).astype('float32')

        train_image_data, test_image_data = np.split(
            sheet_image_data, [int(len(sheet_image_data)*train_split)])
        train_eq_coords, test_eq_coords = np.split(
            sheet_eq_coords, [int(len(sheet_eq_coords)*train_split)]
        )

        # We don't need raw sheet tuple data anymore, unload

        sheets = None
        sheet_image_data = None
        sheet_eq_coords = None

        # Step 3: Train model

        history = self.model.fit(train_image_data, train_eq_coords, epochs=epochs,
                                 validation_data=(test_image_data, test_eq_coords), batch_size=128)

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        test_loss, test_acc = self.model.evaluate(
            test_image_data, test_eq_coords, verbose=2)

        print(test_acc)

    def infer_from_model(self, image_data):
        imdata = np.expand_dims(image_data, axis=0)
        predictions = self.model.predict(imdata)[0]
        coords = []
        for i in range(0, len(predictions), 4):
            coords.append({'x1': predictions[i], 'y1': predictions[i+1],
                           'x2': predictions[i+2], 'y2': predictions[i+3]})

        return coords

    def show_validation(self):
        validation_sheet = EquationSheetGenerator(
            max_equations_per_sheet).generate_sheet()

        rand_test_image = validation_sheet[0]
        rand_test_coords = validation_sheet[1]
        rand_test_image_data = rand_test_image.convert('RGB')
        rand_test_image_data = image.img_to_array(rand_test_image_data)

        fig, ax = plt.subplots()
        ax.imshow(rand_test_image)

        for eq_coord in rand_test_coords:
            xy = (eq_coord['x1'], eq_coord['y1'])
            width = eq_coord['x2'] - xy[0]
            height = eq_coord['y2'] - xy[1]
            ax.add_patch(Rectangle(xy, width, height,
                         fill=False, edgecolor="r"))

        fig, ax = plt.subplots()
        ax.imshow(rand_test_image)

        inferred_coords = self.infer_from_model(rand_test_image_data)

        for eq_coord in inferred_coords:
            xy = (eq_coord['x1'], eq_coord['y1'])
            width = eq_coord['x2'] - xy[0]
            height = eq_coord['y2'] - xy[1]
            ax.add_patch(Rectangle(xy, width, height,
                         fill=False, edgecolor="r"))

        print('Ground truth:')
        print(rand_test_coords)
        print('Inferred:')
        print(inferred_coords)
        print('Differential:')

        differential = [coord_diff(coord, inferred_coord)
                        for coord, inferred_coord in zip(rand_test_coords, inferred_coords)]
        print(list(differential))

        accuracies = [coord_accuracy(coord, inferred_coord)
                      for coord, inferred_coord in zip(rand_test_coords, inferred_coords)]
        print('Accuracy (%):')
        print(accuracies)

        plt.show()
