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
sheet_count = 2000

epochs = 15

# Step 1: Fetch equation sheets

print('Initializing equation sheet image data...')

sheet_images_path = './data/equation-sheet-images'
sheets = EquationSheetGenerator(max_equations_per_sheet).generate_sheets(
    sheet_count, cache_dir=sheet_images_path)

# Step 2: Prepare train and test data

half_sheet_count = int(sheet_count/2)
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


rand_test_image_idx = random.randint(half_sheet_count, sheet_count - 1)
rand_test_image = sheets[rand_test_image_idx][0]
rand_test_image_data = sheet_image_data[rand_test_image_idx]
rand_test_coords = sheets[rand_test_image_idx][1]

train_image_data = sheet_image_data[:half_sheet_count]
train_eq_coords = sheet_eq_coords[:half_sheet_count]

test_image_data = sheet_image_data[half_sheet_count:]
test_eq_coords = sheet_eq_coords[half_sheet_count:]


# We don't need raw sheet tuple data anymore, unload
sheets = None
sheet_image_data = None
sheet_eq_coords = None

train_image_data = np.array(train_image_data).astype('float32')
train_image_data = tf.keras.applications.resnet50.preprocess_input(
    train_image_data)
train_eq_coords = np.array(train_eq_coords).astype('float32')
test_image_data = np.array(test_image_data).astype('float32')
test_image_data = tf.keras.applications.resnet50.preprocess_input(
    test_image_data)
test_eq_coords = np.array(test_eq_coords).astype('float32')

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
    imdata = np.expand_dims(image_data, 0)
    image_data = tf.keras.applications.resnet50.preprocess_input(
        np.expand_dims(image_data, 0))
    predictions = model.predict(image_data)[0]
    coords = []
    # predictions is a len 160 array
    # iterate through and get coords
    for i in range(0, len(predictions), 4):
        coords.append({'x1': predictions[i], 'y1': predictions[i+1],
                      'x2': predictions[i+2], 'y2': predictions[i+3]})

    return coords


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

for eq_coord in inferred_coords:
    xy = (eq_coord['x1'], eq_coord['y1'])
    width = eq_coord['x2'] - xy[0]
    height = eq_coord['y2'] - xy[1]
    ax.add_patch(Rectangle(xy, width, height, fill=False))


plt.show()
