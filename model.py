from pdf2image import convert_from_path
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import random
import math

import PIL
import os
import json

from bounding_rect import BoundingRect
from sheet_generator import SheetGenerator

# _, ax = pyplot.subplots()
# pyplot.imshow(downscaled_image)
# for eq_coord in downscaled_coords:
#     xy = (eq_coord['x1'], eq_coord['y1'])
#     width = eq_coord['x2'] - xy[0]
#     height = eq_coord['y2'] - xy[1]
#     ax.add_patch(Rectangle(xy, width, height, fill=False))

# _, ax = pyplot.subplots()
# new_image, new_coords = SheetGenerator(
#     downscaled_image, downscaled_coords).generate_sheet()
# pyplot.imshow(new_image)
# for eq_coord in new_coords:
#     xy = (eq_coord['x1'], eq_coord['y1'])
#     width = eq_coord['x2'] - xy[0]
#     height = eq_coord['y2'] - xy[1]
#     ax.add_patch(Rectangle(xy, width, height, fill=False))

# pyplot.show()

downscaled_image = convert_from_path('data/sheet1.pdf')[0]
downscaled_image.thumbnail(
    (300, 300), PIL.Image.LANCZOS)
downscaled_coords = json.load(open('data/downscaled_coords.json'))

all_sheets = [(downscaled_image, downscaled_coords)]

generator = SheetGenerator(
    downscaled_image, downscaled_coords)

for i in range(3):
    all_sheets.append(generator.generate_sheet())

# all_sheets is array of 4 300x300 sheets with eq coords
samples = []
for sheet in all_sheets:
    sample = dict()
    greyscale_image = sheet[0].convert('L')
    sample['x'] = list(greyscale_image.getdata(band=0))
    sample['y'] = []
    for coord in sheet[1]:
        sample['y'].append(
            [coord['x1'], coord['y1'], coord['x2'], coord['y2']])
    samples.append(sample)

print(samples)
# for filename in image_files:
#     pil_images = convert_from_path('data/' + filename)
#     original_sheet_image = pil_images[0]
#     original_eq_coords = json.load(open('data/sheet1_coords.json'))
#     fig, ax = pyplot.subplots()
#     ax.imshow(original_sheet_image)
#     for eq_coord in original_eq_coords:
#         eq_image = original_sheet_image.crop(
#             (eq_coord['x1'], eq_coord['y1'],
#              eq_coord['x2'], eq_coord['y2']))
#         xy = (eq_coord['x1'], eq_coord['y1'])
#         width = eq_coord['x2'] - xy[0]
#         height = eq_coord['y2'] - xy[1]
#         ax.add_patch(Rectangle(xy, width, height, fill=False))

#     fig, ax = pyplot.subplots()

#     new_image, new_eq_coords = SheetGenerator(
#         original_sheet_image, original_eq_coords).generate_sheet()
#     pyplot.imshow(new_image)
#     for eq_coord in new_eq_coords:
#         xy = (eq_coord['x1'], eq_coord['y1'])
#         width = eq_coord['x2'] - xy[0]
#         height = eq_coord['y2'] - xy[1]
#         ax.add_patch(Rectangle(xy, width, height, fill=False))
#     pyplot.show()

# there will be varying # of equations per page
