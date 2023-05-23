import random
import PIL
import math
import json
import os
import string
import sys

from .equation_box import EquationBox
from .equation_sheet_decorator import EquationSheetDecorator

from PIL import Image, ImageDraw, ImageFont, ImageOps

from multiprocessing import Pool

RANDOM_TEXT_COUNT_MAX = 15
RANDOM_LINE_COUNT_MAX = 6
RANDOM_ELLIPSE_COUNT_MAX = 12

IMAGE_DIR = './data/images'


def rand_image():
    return Image.open(f'{IMAGE_DIR}/{random.choice(os.listdir(IMAGE_DIR))}')


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


class EquationSheetGenerator:
    def __init__(self, sheet_size=(224, 224), cache_dir=''):
        self.sheet_size = sheet_size
        self.cache_dir = cache_dir

    def generate_sheets(self, sheet_count):
        if len(self.cache_dir) > 0 and self.sheets_cached():
            print('Cached equation sheets found.')
            return self.sheets_from_cache(sheet_count)

        dirty_eq_sheet_count = int(sheet_count * 0.7)
        clean_single_eq_sheet_count = int(sheet_count * 0.1)
        clean_multiple_eq_sheet_count = 0
        blank_clean_sheet_count = int(sheet_count * 0.1)
        rand_img_sheet_count = int(sheet_count * 0.1)

        sheets = []
        should_cache = len(self.cache_dir) > 0
        if should_cache and not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        print('Sheets with one equation and clean background: ',
              clean_single_eq_sheet_count)
        print('Sheets with one equation and dirty background: ',
              dirty_eq_sheet_count)
        print('Sheets with multiple equations and clean background: ',
              clean_multiple_eq_sheet_count)
        print('Sheets with no equations and clean background: ',
              blank_clean_sheet_count)
        print('Sheets with no equations and img background: ',
              rand_img_sheet_count)
        print('Total sheets: ', sheet_count)
        print('Generating equation sheets...')

        sheets = []

        print('Generating clean sheets with equation...')
        sheets.extend([self.clean_sheet_with_equation()
                      for _ in range(clean_single_eq_sheet_count)])
        print('Generating clean sheets with 2 equations...')
        sheets.extend([self.clean_sheet_with_equations()
                      for _ in range(clean_multiple_eq_sheet_count)])
        print('Generating dirty sheets with equation...')
        sheets.extend([self.dirty_sheet_with_equation()
                      for _ in range(dirty_eq_sheet_count)])
        print('Generating rand img sheets...')
        sheets.extend([self.random_image_sheet()
                      for _ in range(rand_img_sheet_count)])
        print('Generating blank clean sheets...')
        sheets.extend([self.blank_sheet_clean()
                      for _ in range(blank_clean_sheet_count)])

        print('Sheets generated. Loading into memory...')

        for idx, sheet in enumerate(sheets):
            if should_cache:
                file_prefix = f'{self.cache_dir}/eq-sheet-{idx}'
                sheet[0].save(f'{file_prefix}.bmp')
                with open(f'{file_prefix}.json', 'w') as coords_file:
                    json.dump(sheet[1].to_eq_coord(), coords_file)

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

        if random.choice([True, False]):
            sheet_image = ImageOps.invert(sheet_image)

        return (sheet_image, eq_box)

    def clean_sheet_with_equations(self):
        sheet_image, eq_box = self.clean_sheet_with_equation()
        if random.choice([True, False]):
            sheet_image = EquationSheetDecorator.add_noise(sheet_image)
        eq_boxes = [eq_box]
        eq_boxes.append(EquationSheetDecorator.add_equation(
            sheet_image, eq_boxes))

        return (sheet_image, eq_box)

    # sheet with colored background and equation image + other misc equation images;
    # includes misc background images
    def dirty_sheet_with_equation(self):
        # sheet_image = self.new_sheet_image(color=random_sheet_color())
        # if random.choice([True, False]):
        #     sheet_image = EquationSheetDecorator.add_noise(sheet_image)

        sheet_image = rand_image()
        sheet_image = sheet_image.resize(self.sheet_size)

        # 50% chance to add random lines
        if random.random() < 0.3:
            line_count = random.randint(1, RANDOM_LINE_COUNT_MAX)
            sheet_image = EquationSheetDecorator.add_lines(
                sheet_image, line_count)

        # # random rectangles of color that are the same size as equation bounding boxes
        # for _ in range(random.randint(1, 4)):
        #     rect_box = EquationSheetDecorator.add_equation(
        #         sheet_image, [eq_box])
        #     color = random.choice(['white', random_color()])
        #     EquationSheetDecorator.add_rectangle(sheet_image, [
        #         rect_box.topLeft[0] + random.randint(1, 4),
        #         rect_box.topLeft[1] + random.randint(1, 4),
        #         rect_box.bottomRight[0] - random.randint(1, 4),
        #         rect_box.bottomRight[1] - random.randint(1, 4),
        #     ], color)

        # # 70% chance to add random text
        # if random.random() < 0.5:
        #     text_count = random.randint(1, RANDOM_TEXT_COUNT_MAX)
        #     sheet_image = EquationSheetDecorator.add_text(
        #         sheet_image, text_count, [eq_box])

        # # 50% chance to add random ellipses
        # if random.random() < 0.3:
        #     ellipse_count = random.randint(1, RANDOM_ELLIPSE_COUNT_MAX)
        #     sheet_image = EquationSheetDecorator.add_ellipses(
        #         sheet_image, ellipse_count, [eq_box])

        # sharpness
        sheet_image = EquationSheetDecorator.adjust_sharpness(
            sheet_image, random.uniform(0.5, 0.7))

        # brightness
        sheet_image = EquationSheetDecorator.adjust_brightness(
            sheet_image, random.uniform(0.05, 0.1))

        # contrast
        sheet_image = EquationSheetDecorator.adjust_contrast(
            sheet_image, random.uniform(0.4, 0.6))

        # color
        sheet_image = EquationSheetDecorator.adjust_color(
            sheet_image, random.uniform(0.5, 0.6))

        eq_box = EquationSheetDecorator.add_equation(sheet_image)

        if random.choice([True, False]):
            sheet_image = sheet_image.convert('RGB')
            sheet_image = ImageOps.invert(sheet_image)

        # rotate sheet
        # rotation_degrees = random.choice([0, 90, 180, 270])
        # sheet_image, eq_boxes = EquationSheetDecorator.rotate_sheet(
        #     sheet_image, [eq_box], rotation_degrees)

        return (sheet_image, eq_box)

    # sheet with no equation; includes misc background images
    def blank_sheet_clean(self):
        sheet_color = 'white'
        if random.random() < 0.5:
            sheet_color = random_color()

        sheet_image = self.new_sheet_image(sheet_color)

        sheet_image = EquationSheetDecorator.add_noise(sheet_image, True)

        if random.random() < 0.5:
            eq_box = EquationSheetDecorator.add_equation(
                sheet_image, [])
            EquationSheetDecorator.add_rectangle(sheet_image, [
                eq_box.topLeft[0],
                eq_box.topLeft[1],
                eq_box.bottomRight[0],
                eq_box.bottomRight[1]
            ], random_color())
        return (sheet_image, EquationBox((0, 0), (0, 0)))

    def random_image_sheet(self):
        rand_img = rand_image()
        rand_img = rand_img.resize(self.sheet_size)

        if random.choice([True, False]):
            rand_img = rand_img.convert('RGB')
            rand_img = ImageOps.invert(rand_img)

        EquationSheetDecorator.add_noise(rand_img, True)

        return (rand_img, EquationBox((0, 0), (0, 0)))

    def blank_sheet(self):
        sheet_image = self.new_sheet_image()

        # random rectangles of color that are the same size as equation bounding boxes
        # keep edges of eq so we get parts equations
        for _ in range(1, 4):
            eq_box = EquationSheetDecorator.add_equation(
                sheet_image, [])
            color = 'white'
            if random.random() < 0.5:
                color = random_color()
            EquationSheetDecorator.add_rectangle(sheet_image, [
                eq_box.topLeft[0] + random.randint(0, 2),
                eq_box.topLeft[1] + random.randint(0, 2),
                eq_box.bottomRight[0] - random.randint(0, 2),
                eq_box.bottomRight[1] - random.randint(0, 2),
            ], color)

        if random.random() < 0.8:
            # random text
            text_count = random.randint(1, RANDOM_TEXT_COUNT_MAX)
            sheet_image = EquationSheetDecorator.add_text(
                sheet_image, text_count, [])

            # random lines
            line_count = random.randint(1, RANDOM_LINE_COUNT_MAX)
            sheet_image = EquationSheetDecorator.add_lines(
                sheet_image, line_count)

            # random ellipses
            ellipse_count = random.randint(1, RANDOM_ELLIPSE_COUNT_MAX)
            sheet_image = EquationSheetDecorator.add_ellipses(
                sheet_image, ellipse_count, [])

        # adjust
        sheet_image = EquationSheetDecorator.adjust_sharpness(
            sheet_image, random.uniform(0.5, 2))

        # brightness
        sheet_image = EquationSheetDecorator.adjust_brightness(
            sheet_image, random.uniform(0.5, 1.5))

        # contrast
        sheet_image = EquationSheetDecorator.adjust_contrast(
            sheet_image, random.uniform(0.6, 1.5))

        # color
        sheet_image = EquationSheetDecorator.adjust_color(
            sheet_image, random.uniform(0.5, 1.5))

        # rotate sheet
        rotation_degrees = random.choice([0, 90, 180, 270])
        sheet_image, eq_boxes = EquationSheetDecorator.rotate_sheet(
            sheet_image, [], rotation_degrees)

        sheet_image = EquationSheetDecorator.add_noise(sheet_image, True)

        if random.choice([True, False]):
            sheet_image = sheet_image.convert('RGB')
            sheet_image = ImageOps.invert(sheet_image)

        return (sheet_image, EquationBox((0, 0), (0, 0)))

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
