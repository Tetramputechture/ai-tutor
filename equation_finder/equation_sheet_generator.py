import random
import PIL
import math
import json
import os

from bounding_rect import BoundingRect
from equation_image_generator import EquationImageGenerator

MIN_EQUATION_WIDTH = 40
MIN_EQUATION_HEIGHT = 10

MAX_EQUATION_WIDTH = 150
MAX_EQUATION_HEIGHT = 50


class EquationSheetGenerator:
    def __init__(self, equation_images, max_equations_per_sheet):
        self.equation_images = equation_images
        self.max_equations_per_sheet = max_equations_per_sheet

    def generate_sheets(self, sheet_count, cache_dir=''):
        if len(cache_dir) > 0 and self.sheets_cached(cache_dir):
            print('Cached equation sheets found.')
            return self.sheets_from_cache(cache_dir)

        print('Generating equation sheets...')
        sheets = []
        should_cache = len(cache_dir) > 0
        if should_cache:
            os.makedirs(cache_dir)

        for idx in range(sheet_count):
            sheet = self.generate_sheet()
            sheets.append(sheet)

            if should_cache:
                file_prefix = f'{cache_dir}/eq-sheet-{idx}'
                sheet[0].save(f'{file_prefix}.bmp')
                with open(f'{file_prefix}.json', 'w') as coords_file:
                    json.dump(sheet[1], coords_file)

        if should_cache:
            print('Equation sheets cached.')

        return sheets

    def generate_sheet(self):
        bounding_rects = []
        eq_coords = []
        sheet_image = PIL.Image.new(
            mode="RGBA", size=(300, 300), color=(tuple(random.choices(range(256), k=3))))
        num_equations = random.randint(1, self.max_equations_per_sheet)
        for equation_image in random.sample(self.equation_images, num_equations):
            # for each equation image, choose random location on sheet
            image_width, image_height = equation_image.size

            iterations = 0
            while iterations < 1000000:
                eq_position = (random.randint(1, 250), random.randint(1, 250))
                eq_bounding_rect = BoundingRect.from_coords(
                    (eq_position[0], eq_position[1]),
                    (eq_position[0] + image_width, eq_position[1] + image_height))
                scale_factor = random.uniform(-0.5, 0.6) + 1
                eq_bounding_rect = eq_bounding_rect.scale(scale_factor)

                collision = False
                if eq_position[0] + image_width > 300 or eq_position[1] + image_height > 300:
                    collision = True

                for rect in bounding_rects:
                    if rect.collision(eq_bounding_rect):
                        collision = True

                if collision:
                    iterations += 1
                else:
                    bounding_rects.append(eq_bounding_rect)

                    equation_image = equation_image.resize(
                        (int(image_width * scale_factor), int(image_height * scale_factor)))
                    image_width, image_height = equation_image.size

                    eq_coords.append({"x1": eq_position[0], "y1": eq_position[1],
                                      "x2": eq_position[0] + image_width, "y2": eq_position[1] + image_height})

                    sheet_image.paste(
                        equation_image, (int(eq_position[0]), int(eq_position[1])), equation_image)
                    break

        # eq_coords must have max equations elements
        # if not, fill with empty coords
        while len(eq_coords) < self.max_equations_per_sheet:
            eq_coords.append({"x1": 0, "y1": 0, "x2": 0, "y2": 0})

        return (sheet_image, eq_coords)

    def sheets_cached(self, cache_dir):
        return os.path.isdir(cache_dir) and len(
            os.listdir(cache_dir)) != 0

    def sheets_from_cache(self, cache_dir):
        sheets = []
        for filename in os.listdir(cache_dir):
            file_prefix, file_ext = os.path.splitext(filename)
            if file_ext == '.bmp':
                image_file = os.path.join(cache_dir, filename)
                coords_file = os.path.join(
                    cache_dir, f'{file_prefix}.json')
                coords_file_data = open(coords_file)
                if os.path.isfile(image_file):
                    sheet_image = PIL.Image.open(image_file)
                    sheet_coords = json.load(coords_file_data)
                    sheets.append((sheet_image, sheet_coords))

        return sheets
