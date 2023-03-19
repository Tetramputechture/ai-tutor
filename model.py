from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import random
import math
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models

import PIL
import os
import json
import sys

from bounding_rect import BoundingRect
from equation_image_generator import EquationImageGenerator
from equation_sheet_generator import EquationSheetGenerator

equation_count = 200
max_equations_per_sheet = 2
sheet_count = 1000

epochs = 10

# Step 1: Fetch cached equation images, or generate them if they are not available

print('Initializing equation image data...')

equation_images_path = 'data/equation-images'
equation_images_cached = os.path.isdir(equation_images_path) and len(
    os.listdir(equation_images_path)) != 0


if equation_images_cached:
    print('Cached equation images found. Loading images...')
    equation_images = []
    for filename in os.listdir(equation_images_path):
        image_file = os.path.join(equation_images_path, filename)
        if os.path.isfile(image_file):
            equation_image = PIL.Image.open(image_file)
            equation_images.append(equation_image)
else:
    print('Cached equation images not found. Generating equation images...')
    equation_images = EquationImageGenerator().generate_equation_images(equation_count)
    os.makedirs(equation_images_path)
    for idx, equation_image in enumerate(equation_images):
        filename = f'{equation_images_path}/eq-{idx}.png'
        equation_image.save(filename)

print('Equation images loaded.')

# Step 2: Fetch cached equation sheets, or generate them if they are not available

print('Initializing equation sheet image data...')

sheet_images_path = 'data/equation-sheet-images'
sheet_images_cached = os.path.isdir(sheet_images_path) and len(
    os.listdir(sheet_images_path)) != 0

if sheet_images_cached:
    print('Cached equation sheet images found. Loading images...')
    sheets = []
    for filename in os.listdir(sheet_images_path):
        file_prefix, file_ext = os.path.splitext(filename)
        if file_ext == '.bmp':
            image_file = os.path.join(sheet_images_path, filename)
            coords_file = os.path.join(
                sheet_images_path, f'{file_prefix}.json')
            coords_file_data = open(coords_file)
            if os.path.isfile(image_file):
                sheet_image = PIL.Image.open(image_file)
                sheet_coords = json.load(coords_file_data)
                sheets.append((sheet_image, sheet_coords))
else:
    print('Cached equation sheet images not found. Generating equation sheet images...')
    sheets = EquationSheetGenerator(
        equation_images, max_equations_per_sheet).generate_sheet_images(sheet_count)
    os.makedirs(sheet_images_path)
    for idx, sheet in enumerate(sheets):
        file_prefix = f'{sheet_images_path}/eq-sheet-{idx}'
        # sheet is a tuple (image, eq_coords)
        sheet[0].save(f'{file_prefix}.bmp')
        with open(f'{file_prefix}.json', 'w') as coords_file:
            json.dump(sheet[1], coords_file)


# Step 3: Prepare train and test data by splitting sheet array into two
# train_sheets = sheets[:int(sheet_count/2)]
# test_sheets = sheets[int(sheet_count/2):]

half_sheet_count = int(sheet_count/2)
sheet_images = []
sheet_coords = []

for sheet in sheets:
    image, eq_coords = sheet
    sample = dict()
    greyscale_image = image.convert('RGB')
    im_arr = np.array(greyscale_image).astype('float32')
    sheet_images.append(im_arr)
    coords = []
    for coord in eq_coords:
        coords.extend(
            [coord['x1'], coord['y1'], coord['x2'], coord['y2']])
    sheet_coords.append(coords)

train_images = sheet_images[:half_sheet_count]
train_coords = sheet_coords[:half_sheet_count]

test_images = sheet_images[half_sheet_count:]
test_coords = sheet_coords[half_sheet_count:]

train_images = np.array(train_images).astype('float32')
train_coords = np.array(train_coords).astype('float32')
test_images = np.array(test_images).astype('float32')
test_coords = np.array(test_coords).astype('float32')

# Step 4: Train model

base_model = tf.keras.applications.resnet.ResNet50(
    include_top=False,
    input_shape=(300, 300, 3)
)
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(max_equations_per_sheet * 4, activation='relu'))

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-26:]:
    layer.trainable = True

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

history = model.fit(train_images, train_coords, epochs=epochs,
                    validation_data=(test_images, test_coords), batch_size=64)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_coords, verbose=2)

print(test_acc)


def infer_from_model(image_data):
    predictions = model.predict(np.expand_dims(image_data, 0))[0]
    coords = []
    # predictions is a len 160 array
    # iterate through and get coords
    for i in range(0, len(predictions), 4):
        coords.append({'x1': predictions[i], 'y1': predictions[i+1],
                      'x2': predictions[i+2], 'y2': predictions[i+3]})

    return coords


rand_test_image_idx = random.randint(0, half_sheet_count - 1)
rand_test_image_data = test_images[rand_test_image_idx]
rand_test_image = sheets[half_sheet_count + rand_test_image_idx - 1][0]
rand_test_coords = sheets[half_sheet_count + rand_test_image_idx - 1][1]

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
for eq_coord in inferred_coords:
    xy = (eq_coord['x1'], eq_coord['y1'])
    width = eq_coord['x2'] - xy[0]
    height = eq_coord['y2'] - xy[1]
    ax.add_patch(Rectangle(xy, width, height, fill=False))


plt.show()
