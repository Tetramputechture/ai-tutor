import random
import PIL
import math
import json
import os
import string
import sys

from .equation_box import EquationBox
from .equation_image_generator import EquationImageGenerator
from .equation_sheet_decorator import EquationSheetDecorator

from PIL import Image, ImageDraw, ImageFont, ImageOps

from multiprocessing import Pool

RANDOM_TEXT_COUNT_MAX = 15
RANDOM_LINE_COUNT_MAX = 6
RANDOM_ELLIPSE_COUNT_MAX = 12


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


def random_font():
    return random.choice([
        './assets/fonts/ArefRuqaa-Regular.ttf',
        './assets/fonts/BungeeColor-Regular_colr_Windows.ttf',
        './assets/fonts/FreeMono.ttf',
        './assets/fonts/KhmerOSBattambang-Regular.ttf',
        './assets/fonts/NotoSans-Regular.ttf',
        './assets/fonts/OpenSansCondensed-LightItalic.ttf'
    ])


class EquationSheetGenerator:
    def __init__(self, sheet_size=(227, 227), cache_dir=''):
        self.sheet_size = sheet_size
        self.cache_dir = cache_dir

    def generate_sheets(self, sheet_count):
        if len(self.cache_dir) > 0 and self.sheets_cached():
            print('Cached equation sheets found.')
            return self.sheets_from_cache(sheet_count)

        print('Generating equation sheets...')

        eq_sheet_count = int(sheet_count * 0.5)
        dirty_eq_sheet_count = int(eq_sheet_count * 0.7)
        clean_single_eq_sheet_count = int(
            (eq_sheet_count - dirty_eq_sheet_count) * 0.8)
        clean_multiple_eq_sheet_count = int(
            (eq_sheet_count - dirty_eq_sheet_count) * 0.2)
        blank_sheet_count = sheet_count - clean_single_eq_sheet_count - \
            clean_multiple_eq_sheet_count - dirty_eq_sheet_count
        blank_dirty_sheet_count = int(blank_sheet_count / 2)
        blank_clean_sheet_count = blank_dirty_sheet_count

        sheets = []
        should_cache = len(self.cache_dir) > 0
        if should_cache and not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        print('Sheets with one equation and clean background: ',
              clean_single_eq_sheet_count)
        print('Sheets with multiple equations and clean background: ',
              clean_multiple_eq_sheet_count)
        print('Sheets with one equation and dirty background: ',
              dirty_eq_sheet_count)
        print('Sheets no with equation and clean background: ',
              blank_clean_sheet_count)
        print('Sheets no with equation and dirty background: ',
              blank_dirty_sheet_count)
        print('Total sheets with equation: ', eq_sheet_count)
        print('Total sheets: ', sheet_count)

        sheet_gen_results = []
        # generate 100 sheets at a time
        for i in range(0, clean_single_eq_sheet_count, 200):
            with Pool(processes=8) as pool:
                for _ in range(min(200, clean_single_eq_sheet_count - i)):
                    sheet_gen_results.append(pool.apply_async(
                        self.clean_sheet_with_equation))
                pool.close()
                pool.join()

        for i in range(0, clean_multiple_eq_sheet_count, 200):
            with Pool(processes=8) as pool:
                for _ in range(min(200, clean_multiple_eq_sheet_count - i)):
                    sheet_gen_results.append(pool.apply_async(
                        self.clean_sheet_with_equations))
                pool.close()
                pool.join()

        for i in range(0, dirty_eq_sheet_count, 200):
            with Pool(processes=8)as pool:
                for _ in range(min(200, dirty_eq_sheet_count - i)):
                    sheet_gen_results.append(pool.apply_async(
                        self.dirty_sheet_with_equation))
                pool.close()
                pool.join()

        for i in range(0, blank_dirty_sheet_count, 200):
            with Pool(processes=8) as pool:
                for _ in range(min(200, blank_dirty_sheet_count - i)):
                    sheet_gen_results.append(pool.apply_async(
                        self.blank_sheet
                    ))
                pool.close()
                pool.join()

        for i in range(0, blank_clean_sheet_count, 200):
            with Pool(processes=8) as pool:
                for _ in range(min(200, blank_clean_sheet_count - i)):
                    sheet_gen_results.append(pool.apply_async(
                        self.blank_sheet_clean
                    ))
                pool.close()
                pool.join()

        print('Sheets generated. Loading into memory...')

        for idx, sheet_gen_result in enumerate(sheet_gen_results):
            sheet = sheet_gen_result.get()
            sheets.append(sheet)

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
            mode="RGBA", size=self.sheet_size, color=color)

    # sheet with white background and 1 equation image; no misc background images
    def clean_sheet_with_equation(self, noise=True):
        sheet_image = self.new_sheet_image(color=random_sheet_color())
        if noise:
            sheet_image = EquationSheetDecorator.add_noise(sheet_image)
        eq_box = EquationSheetDecorator.add_equation(sheet_image)

        return (sheet_image, eq_box)

    def clean_sheet_with_equations(self):
        sheet_image, eq_box = self.clean_sheet_with_equation()
        sheet_image = EquationSheetDecorator.add_noise(sheet_image)
        eq_boxes = [eq_box]
        eq_boxes.append(EquationSheetDecorator.add_equation(
            sheet_image, eq_boxes))

        return (sheet_image, eq_box)

    # sheet with colored background and equation image + other misc equation images;
    # includes misc background images
    def dirty_sheet_with_equation(self):
        sheet_image = self.new_sheet_image(color=random_sheet_color())
        sheet_image = EquationSheetDecorator.add_noise(sheet_image)

        # 50% chance to add random lines
        if random.random() < 0.3:
            line_count = random.randint(1, RANDOM_LINE_COUNT_MAX)
            sheet_image = EquationSheetDecorator.add_lines(
                sheet_image, line_count)

        eq_box = EquationSheetDecorator.add_equation(sheet_image)

        # random rectangles of color that are the same size as equation bounding boxes
        for _ in range(random.randint(1, 4)):
            rect_box = EquationSheetDecorator.add_equation(
                sheet_image, [eq_box])
            color = random.choice(['white', random_color()])
            EquationSheetDecorator.add_rectangle(sheet_image, [
                rect_box.topLeft[0] + random.randint(2, 12),
                rect_box.topLeft[1] + random.randint(2, 12),
                rect_box.bottomRight[0] - random.randint(2, 12),
                rect_box.bottomRight[1] - random.randint(2, 12)
            ], color)

        # 70% chance to add random text
        if random.random() < 0.5:
            text_count = random.randint(1, RANDOM_TEXT_COUNT_MAX)
            sheet_image = EquationSheetDecorator.add_text(
                sheet_image, text_count, [eq_box])

        # 50% chance to add random ellipses
        if random.random() < 0.3:
            ellipse_count = random.randint(1, RANDOM_ELLIPSE_COUNT_MAX)
            sheet_image = EquationSheetDecorator.add_ellipses(
                sheet_image, ellipse_count, [eq_box])

        # sharpness
        sheet_image = EquationSheetDecorator.adjust_sharpness(
            sheet_image, random.uniform(0.2, 2))

        # brightness
        sheet_image = EquationSheetDecorator.adjust_brightness(
            sheet_image, random.uniform(0.8, 1.2))

        # contrast
        sheet_image = EquationSheetDecorator.adjust_contrast(
            sheet_image, random.uniform(0.9, 1.1))

        # color
        sheet_image = EquationSheetDecorator.adjust_color(
            sheet_image, random.uniform(0.9, 1.1))

        # 0.5 chance to invert color
        if random.random() < 0.5:
            sheet_image = EquationSheetDecorator.invert_color(sheet_image)

        # rotate sheet
        rotation_degrees = random.choice([0, 90, 180, 270])
        sheet_image, eq_boxes = EquationSheetDecorator.rotate_sheet(
            sheet_image, [eq_box], rotation_degrees)

        return (sheet_image, eq_boxes[0])

    # sheet with no equation; includes misc background images
    def blank_sheet_clean(self):
        sheet_color = 'white'
        if random.random() < 0.5:
            sheet_color = random_color()

        sheet_image = self.new_sheet_image(sheet_color)

        if random.random() < 0.5:
            sheet_image = EquationSheetDecorator.add_noise(sheet_image)

        if random.random() < 0.5:
            eq_box = EquationSheetDecorator.add_equation(
                sheet_image, [])
            EquationSheetDecorator.add_rectangle(sheet_image, [
                eq_box.topLeft[0],
                eq_box.topLeft[1],
                eq_box.bottomRight[0],
                eq_box.bottomRight[1]
            ], 'white')
        return (sheet_image, EquationBox((0, 0), (0, 0)))

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
                eq_box.topLeft[0] + random.randint(6, 12),
                eq_box.topLeft[1] + random.randint(6, 12),
                eq_box.bottomRight[0] - random.randint(6, 12),
                eq_box.bottomRight[1] - random.randint(6, 12)
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
            sheet_image, random.uniform(0.2, 2))

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

        sheet_image = EquationSheetDecorator.add_noise(sheet_image)

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
        sheets = []
        with Pool(processes=4) as pool:
            sheets = pool.map(
                self.sheet_from_file, bmp_files, chunksize=50)
            pool.close()
            pool.join()

        return sheets[:sheet_count]
