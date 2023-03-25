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

from bounding_rect import BoundingRect
from equation_image_generator import EquationImageGenerator
from equation_sheet_generator import EquationSheetGenerator


from resnet_model import ResnetModel
from conv_model import ConvModel

max_equations_per_sheet = 1
sheet_count = 1000

epochs = 15

train_split = 0.5

# Step 1: Fetch equation sheets

print('Initializing equation sheet image data...')

sheet_images_path = './data/equation-sheet-images'
sheets = EquationSheetGenerator(max_equations_per_sheet).generate_sheets(
    sheet_count, cache_dir=sheet_images_path)

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

model = ResnetModel().create_model()
# model = ConvModel().create_model()

history = model.fit(train_image_data, train_eq_coords, epochs=epochs,
                    validation_data=(test_image_data, test_eq_coords), batch_size=64)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(
    test_image_data, test_eq_coords, verbose=2)

print(test_acc)


def infer_from_model(image_data):
    imdata = np.expand_dims(image_data, axis=0)

    # image_data = tf.keras.applications.resnet50.preprocess_input(
    #     np.expand_dims(image_data, 0))
    predictions = model.predict(imdata)[0]
    coords = []
    # predictions is a len 160 array
    # iterate through and get coords
    for i in range(0, len(predictions), 4):
        coords.append({'x1': predictions[i], 'y1': predictions[i+1],
                      'x2': predictions[i+2], 'y2': predictions[i+3]})

    return coords


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
    ax.add_patch(Rectangle(xy, width, height, fill=False))

fig, ax = plt.subplots()
ax.imshow(rand_test_image)

inferred_coords = infer_from_model(rand_test_image_data)

for eq_coord in inferred_coords:
    xy = (eq_coord['x1'], eq_coord['y1'])
    width = eq_coord['x2'] - xy[0]
    height = eq_coord['y2'] - xy[1]
    ax.add_patch(Rectangle(xy, width, height, fill=False))

print('Ground truth:')
print(rand_test_coords)
print('Inferred:')
print(inferred_coords)
print('Differential:')


def diff(idx, coord):
    return {
        'x1': abs(coord['x1'] - inferred_coords[idx]['x1']),
        'y1': abs(coord['y1'] - inferred_coords[idx]['y1']),
        'x2': abs(coord['x2'] - inferred_coords[idx]['x2']),
        'y2': abs(coord['y2'] - inferred_coords[idx]['y2'])
    }


differential = [diff(idx, coord) for idx, coord in enumerate(rand_test_coords)]
print(list(differential))


def accuracy(idx, coord):
    return {
        'x1': (differential[idx]['x1'] / coord['x1']) * 100,
        'y1': (differential[idx]['y1'] / coord['y1']) * 100,
        'x2': (differential[idx]['x2'] / coord['x2']) * 100,
        'y2': (differential[idx]['y2'] / coord['y2']) * 100
    }


accuracies = [accuracy(idx, coord)
              for idx, coord in enumerate(rand_test_coords)]
print('Accuracy (%):')
print(accuracies)

plt.show()
