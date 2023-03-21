import random
import PIL
import math
import json
import os
import string

from bounding_rect import BoundingRect
from equation_image_generator import EquationImageGenerator

from PIL import Image, ImageDraw, ImageFont

MIN_EQUATION_WIDTH = 40
MIN_EQUATION_HEIGHT = 10

MAX_EQUATION_WIDTH = 150
MAX_EQUATION_HEIGHT = 50

RANDOM_TEXT_COUNT_MAX = 15

RANDOM_LINE_COUNT_MAX = 20

RANDOM_ELLIPSE_COUNT_MAX = 10


def random_text():
    text_len = random.randint(3, 12)
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
                   for _ in range(text_len))


def random_color():
    return tuple(random.choices(range(256), k=3))


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
        if should_cache and not os.path.isdir(cache_dir):
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
        sheet_image = Image.new(
            mode="RGBA", size=(300, 300), color=(tuple(random.choices(range(256), k=3))))
        num_equations = random.randint(1, self.max_equations_per_sheet)
        for equation_image in random.sample(self.equation_images, num_equations):
            # for each equation image, choose random location on sheet
            image_width, image_height = equation_image.size

            iterations = 0
            while iterations < 1000000:
                eq_position = (random.randint(1, 250), random.randint(1, 250))
                eq_bounding_rect = BoundingRect.from_coords(
                    eq_position,
                    (eq_position[0] + image_width,
                     eq_position[1] + image_height)
                )
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

        while len(eq_coords) < self.max_equations_per_sheet:
            eq_coords.append({"x1": 0, "y1": 0, "x2": 0, "y2": 0})

        # add misc other text
        for i in range(random.randint(3, RANDOM_TEXT_COUNT_MAX)):
            iterations = 0
            while iterations < 1000000:
                text_image = Image.new(
                    'RGBA', sheet_image.size, (255, 255, 255, 0))
                fnt = ImageFont.truetype(random_font(), random.randint(6, 14))

                image_draw_ctx = ImageDraw.Draw(text_image)

                text = random_text()
                text_position = (random.randint(1, 250),
                                 random.randint(1, 250))
                image_draw_ctx.text(text_position, text,
                                    font=fnt, fill=(*random_color(), random.randint(150, 255)))

                text_width, text_height = image_draw_ctx.textsize(
                    text, font=fnt, spacing=8)

                text_bounding_rect = BoundingRect.from_coords(
                    text_position,
                    (text_position[0] + text_width,
                     text_position[1] + text_height)
                )

                collision = False
                for rect in bounding_rects:
                    collision = rect.collision(text_bounding_rect)

                if collision:
                    iterations += 1
                else:
                    sheet_image = Image.alpha_composite(
                        sheet_image, text_image)
                    break

        # add random lines
        sheet_image_draw_ctx = ImageDraw.Draw(sheet_image)
        for i in range(random.randint(0, RANDOM_LINE_COUNT_MAX)):
            line_position = (random.randint(15, 250), random.randint(15, 250))
            line_size = (random.randint(50, 200), random.randint(50, 200))
            sheet_image_draw_ctx.line(
                line_position + line_size, fill=(*random_color(), random.randint(150, 255)), width=random.randint(1, 2))

        # add ellipses
        for i in range(random.randint(0, RANDOM_ELLIPSE_COUNT_MAX)):
            iterations = 0
            while iterations < 1000000:
                ellipse_position = (random.randint(1, 250),
                                    random.randint(1, 250))
                ellipse_width, ellipse_height = (
                    random.randint(5, 20), random.randint(5, 20))

                ellipse_bounding_rect = BoundingRect.from_coords(
                    ellipse_position,
                    (ellipse_position[0] + ellipse_width,
                     ellipse_position[1] + ellipse_height)
                )

                collision = False

                for rect in bounding_rects:
                    if rect.collision(ellipse_bounding_rect):
                        collision = True

                if collision:
                    iterations += 1
                else:
                    fill_color = (0, 0, 0, 0)
                    if random.random() > 0.5:
                        fill_color = random_color()
                    sheet_image_draw_ctx.ellipse(
                        [ellipse_position, (ellipse_position[0] + ellipse_width,
                                            ellipse_position[1] + ellipse_height)],
                        fill=fill_color, outline=random_color(), width=1
                    )
                    break

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
                    sheet_image = Image.open(image_file)
                    sheet_coords = json.load(coords_file_data)
                    sheets.append((sheet_image, sheet_coords))

        return sheets