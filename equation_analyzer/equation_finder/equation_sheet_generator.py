import random
import PIL
import math
import json
import os
import string
import sys
import csv

from .equation_box import EquationBox
from .equation_sheet_decorator import EquationSheetDecorator

from PIL import Image, ImageDraw, ImageFont, ImageOps

from multiprocessing import Pool

RANDOM_TEXT_COUNT_MAX = 15
RANDOM_LINE_COUNT_MAX = 6
RANDOM_ELLIPSE_COUNT_MAX = 12

IMAGE_DIR = './data/images'
CUSTOM_EQ_IMAGE_DIR = './data/images_custom'


def rand_image():
    return Image.open(f'{IMAGE_DIR}/{random.choice(os.listdir(IMAGE_DIR))}')


def rand_custom_eq_image():
    jpg_files = [f for f in os.listdir(
        CUSTOM_EQ_IMAGE_DIR) if f.endswith('.jpg')]
    rand_file = random.choice(jpg_files)
    return (
        Image.open(
            f'{CUSTOM_EQ_IMAGE_DIR}/{rand_file}'),
        rand_file.split('.')[0]
    )


def random_color():
    return tuple(random.choices(range(256), k=3))


def random_sheet_color():
    return random.choice([
        'aliceblue',
        'whitesmoke',
        'white',
        'lightgray',
        'linen',
        'lavender',
        'snow',
        'bisque'
    ])


DEBUG = False


class EquationSheetGenerator:
    def __init__(self, sheet_size=(224, 224), cache_dir=''):
        self.sheet_size = sheet_size
        self.cache_dir = cache_dir
        self.custom_eq_boxes = []
        with open('./data/images_custom/eq_boxes.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for line in reader:
                self.custom_eq_boxes.append(line)

    def custom_image_eq_box(self, img_idx):
        box = [box for box in self.custom_eq_boxes if box['image'] == img_idx][0]
        return EquationBox((box['x1'], box['y1']), (box['x2'], box['y2']))

    def generate_sheets(self, sheet_count):
        if len(self.cache_dir) > 0 and self.sheets_cached():
            print('Cached equation sheets found.')
            return self.sheets_from_cache(sheet_count)

        dirty_eq_sheet_count = int(sheet_count * 0.1)
        dirty_eq_img_sheet_count = int(sheet_count * 0.6)
        clean_single_eq_sheet_count = int(sheet_count * 0.05)
        rand_img_sheet_count = int(sheet_count * 0.1)
        custom_sheet_count = int(sheet_count * 0.15)

        sheets = []
        should_cache = len(self.cache_dir) > 0
        if should_cache and not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        print('Sheets with one equation and clean background: ',
              clean_single_eq_sheet_count)
        print('Sheets with one equation and dirty background: ',
              dirty_eq_sheet_count)
        print('Sheets with one equation on sheet and dirty background: ',
              dirty_eq_img_sheet_count)
        print('Sheets with no equations and img background: ',
              rand_img_sheet_count)
        print('Sheets with custom equation and img background: ',
              custom_sheet_count)
        print('Total sheets: ', sheet_count)
        print('Generating equation sheets...')

        sheets = []

        print('Generating clean sheets with equation...')
        sheets.extend([self.clean_sheet_with_equation()
                      for _ in range(clean_single_eq_sheet_count)])
        print('Generating dirty sheets with equation...')
        sheets.extend([self.dirty_sheet_with_equation()
                      for _ in range(dirty_eq_sheet_count)])
        print('Generating dirty sheets with imagenet equation...')
        sheets.extend([self.dirty_sheet_with_equation(True)
                      for _ in range(dirty_eq_img_sheet_count)])
        print('Generating custom sheets...')
        sheets.extend([self.custom_sheet()
                      for _ in range(custom_sheet_count)])
        print('Generating rand img sheets...')
        sheets.extend([self.random_image_sheet()
                      for _ in range(rand_img_sheet_count)])

        print('Sheets generated. Loading into memory...')

        for idx, sheet in enumerate(sheets):
            eq_box = sheet[1]
            if DEBUG:
                ImageDraw.Draw(sheet[0]).rectangle(
                    [eq_box.topLeft[0], eq_box.topLeft[1], eq_box.bottomRight[0], eq_box.bottomRight[1]], outline='red')
            if should_cache:
                file_prefix = f'{self.cache_dir}/eq-sheet-{idx}'
                sheet[0].save(f'{file_prefix}.bmp')
                with open(f'{file_prefix}.json', 'w') as coords_file:
                    json.dump(eq_box.to_eq_coord(), coords_file)

        if should_cache:
            print('Equation sheets cached.')

        return sheets

    def new_sheet_image(self, color='white'):
        return Image.new(
            mode="RGB", size=self.sheet_size, color=color)

    # sheet with white background and 1 equation image; no misc background images
    def clean_sheet_with_equation(self, noise=True):
        sheet_image = self.new_sheet_image(color=random_sheet_color())
        if noise and random.choice([True, False]):
            sheet_image = EquationSheetDecorator.add_noise(sheet_image)
        eq_box = EquationSheetDecorator.add_equation(sheet_image)

        rotation_degrees = random.choice([0, 90, 180, 270])
        sheet_image, eq_boxes = EquationSheetDecorator.rotate_sheet(
            sheet_image, [eq_box], rotation_degrees)

        if random.choice([True, False]):
            sheet_image = ImageOps.invert(sheet_image)

        return (sheet_image, eq_boxes[0])

    # sheet with colored background and equation image + other misc equation images;
    # includes misc background images
    def dirty_sheet_with_equation(self, include_bg=False):
        sheet_image = rand_image()
        sheet_image = sheet_image.resize(self.sheet_size)

        if include_bg:
            fg_sheet = self.new_sheet_image()
            unscaled_eq_box = EquationSheetDecorator.add_equation(fg_sheet)

            fg_sheet_scale = (random.uniform(0.6, 0.8),
                              random.uniform(0.6, 0.8))

            fg_sheet = fg_sheet.resize((int(self.sheet_size[0] * fg_sheet_scale[0]),
                                        int(self.sheet_size[1] * fg_sheet_scale[1])))

            fg_sheet_size = fg_sheet.size

            fg_pos = (random.randint(0, self.sheet_size[0] - fg_sheet_size[0]),
                      random.randint(0, self.sheet_size[1] - fg_sheet_size[1]))

            eq_box = unscaled_eq_box.scale(fg_sheet_scale)
            eq_box = eq_box.shift(fg_pos)

            sheet_image.paste(fg_sheet, fg_pos)
        else:
            # sharpness
            sheet_image = EquationSheetDecorator.adjust_sharpness(
                sheet_image, random.uniform(0.5, 0.7))

            # brightness
            sheet_image = EquationSheetDecorator.adjust_brightness(
                sheet_image, random.uniform(0.8, 1.2))

            # contrast
            sheet_image = EquationSheetDecorator.adjust_contrast(
                sheet_image, random.uniform(0.4, 0.6))

            # color
            sheet_image = EquationSheetDecorator.adjust_color(
                sheet_image, random.uniform(0.5, 1.2))

            eq_box = EquationSheetDecorator.add_equation(sheet_image, [], True)

        if random.choice([True, False]):
            sheet_image = sheet_image.convert('RGB')
            sheet_image = ImageOps.invert(sheet_image)

        # rotate sheet
        rotation_degrees = random.choice([0, 90, 180, 270])
        sheet_image, eq_boxes = EquationSheetDecorator.rotate_sheet(
            sheet_image, [eq_box], rotation_degrees)

        eq_box = eq_boxes[0]

        return (sheet_image, eq_box)

    def random_image_sheet(self):
        rand_img = rand_image()
        rand_img = rand_img.resize(self.sheet_size)

        if random.choice([True, False]):
            rand_img = rand_img.convert('RGB')
            rand_img = ImageOps.invert(rand_img)

        rotation_degrees = random.randint(0, 259)
        rand_img, eq_boxes = EquationSheetDecorator.rotate_sheet(
            rand_img, [], rotation_degrees)

        EquationSheetDecorator.add_noise(rand_img, True)

        return (rand_img, EquationBox((0, 0), (0, 0)))

    def rand_rotation_angle(self):
        return random.randint(-45, 45)

    def custom_sheet(self):
        img, idx = rand_custom_eq_image()
        eq_box = self.custom_image_eq_box(idx)

        eq_box_is_zero = eq_box.is_zero()

        rotation_degrees = self.rand_rotation_angle()
        sheet_image, eq_boxes = EquationSheetDecorator.rotate_sheet(
            img, [eq_box], rotation_degrees)

        if random.choice([True, False]):
            sheet_image = sheet_image.convert('RGB')
            sheet_image = ImageOps.invert(sheet_image)

        full_img = rand_image()
        full_img = full_img.resize(self.sheet_size)

        random_scale = (random.uniform(0.9, 1), random.uniform(0.9, 1))
        new_size = (int(self.sheet_size[0] * random_scale[0]),
                    int(self.sheet_size[1] * random_scale[1]))
        sheet_image = sheet_image.resize(new_size)

        sheet_location = (
            (random.randint(0, self.sheet_size[0] - new_size[0]),
             random.randint(0, self.sheet_size[1] - new_size[1]))
        )
        full_img.paste(sheet_image, sheet_location)
        scaled_eq_box = eq_boxes[0]
        if not scaled_eq_box.is_zero():
            scaled_eq_box = eq_boxes[0].scale(random_scale)
            scaled_eq_box = scaled_eq_box.shift(sheet_location)

        rotation_degrees = random.choice([0, 90, 180, 270])
        full_img, eq_boxes = EquationSheetDecorator.rotate_sheet(
            full_img, [scaled_eq_box], rotation_degrees)

        eq_box = eq_boxes[0]

        if eq_box_is_zero:
            return (full_img, EquationBox((0, 0), (0, 0)))

        return (full_img, eq_box)

    def sheets_cached(self):
        return os.path.isdir(self.cache_dir) and len(
            os.listdir(self.cache_dir)) != 0

    def sheet_from_file(self, filename):
        file_prefix = os.path.splitext(filename)[0]
        image_file = os.path.join(self.cache_dir, filename)
        coords_data_file = os.path.join(
            self.cache_dir, f'{file_prefix}.json')
        if os.path.isfile(image_file):
            sheet_image = Image.open(image_file)
            with open(coords_data_file) as coords_file:
                sheet_coord = json.loads(coords_file.read())

        return (sheet_image, EquationBox.from_eq_coord(sheet_coord))

    def sheets_from_cache(self, sheet_count):
        bmp_files = [f for f in os.listdir(
            self.cache_dir) if f.endswith('.bmp')]
        bmp_files = bmp_files[:sheet_count]
        sheets = []
        with Pool(processes=4) as pool:
            sheets = pool.map(
                self.sheet_from_file, bmp_files, chunksize=50)
            pool.close()
            pool.join()

        return sheets
