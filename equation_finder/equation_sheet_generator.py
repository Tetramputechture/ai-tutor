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

RANDOM_TEXT_COUNT_MAX = 35

RANDOM_LINE_COUNT_MAX = 20

RANDOM_ELLIPSE_COUNT_MAX = 30


def random_text():
    text_len = random.randint(3, 12)
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
                   for _ in range(text_len))


def random_color():
    return tuple(random.choices(range(256), k=3))


def random_sheet_color():
    return random.choice([
        'aliceblue',
        'fuchsia',
        'khaki',
        'white',
        'lawngreen',
        'lightpink',
        'lightslategray',
        'linen',
        'mediumblue',
        'palegreen',
        'snow',
        'skyblue',
        'yellowgreen',
        'seashell'
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
    def __init__(self, max_equations_per_sheet):
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
            mode="RGBA", size=(300, 300), color=random_sheet_color())
        num_equations = random.randint(1, self.max_equations_per_sheet)
        sheet_image_draw_ctx = ImageDraw.Draw(sheet_image)
        eq_im_generator = EquationImageGenerator()
        for i in range(num_equations):
            equation_image = eq_im_generator.generate_equation_image()
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
                scale_factor = random.uniform(0.1, 0.6) + 1
                eq_bounding_rect = eq_bounding_rect.scale(scale_factor)

                collision = False
                if eq_position[0] + eq_bounding_rect.width > 300 or eq_position[1] + eq_bounding_rect.height > 300:
                    collision = True

                for rect in bounding_rects:
                    if rect.collision(eq_bounding_rect):
                        collision = True
                        break

                if collision:
                    iterations += 1
                else:

                    equation_image = equation_image.resize(
                        (int(image_width * scale_factor), int(image_height * scale_factor)))
                    image_width, image_height = equation_image.size

                    eq_coords.append({"x1": eq_position[0], "y1": eq_position[1],
                                      "x2": eq_position[0] + image_width, "y2": eq_position[1] + image_height})

                    bounding_rects.append(eq_bounding_rect)

                    sheet_image.paste(
                        equation_image, (int(eq_position[0]), int(eq_position[1])), equation_image)
                    break

        while len(eq_coords) < self.max_equations_per_sheet:
            eq_coords.append({"x1": 0, "y1": 0, "x2": 0, "y2": 0})

        # add misc other text
        for i in range(random.randint(5, RANDOM_TEXT_COUNT_MAX)):
            iterations = 0
            text_image = Image.new(
                'RGBA', sheet_image.size, (255, 255, 255, 0))
            fnt = ImageFont.truetype(random_font(), random.randint(6, 14))
            image_draw_ctx = ImageDraw.Draw(text_image)
            text = random_text()
            while iterations < 1000000:
                text_position = (random.randint(1, 250),
                                 random.randint(1, 250))
                image_draw_ctx.text(text_position, text,
                                    font=fnt, fill=(*random_color(), 0))

                text_width, text_height = image_draw_ctx.textsize(
                    text, font=fnt, spacing=4)

                text_bounding_rect = BoundingRect.from_coords(
                    text_position,
                    (text_position[0] + text_width,
                     text_position[1] + text_height)
                )

                collision = False
                for rect in bounding_rects:
                    if rect.collision(text_bounding_rect):
                        collision = True
                        break

                if collision:
                    iterations += 1
                else:
                    image_draw_ctx.text(text_position, text,
                                        font=fnt, fill=(*random_color(), random.randint(150, 255)))

                    sheet_image = Image.alpha_composite(
                        sheet_image, text_image)
                    break

        # add random lines
        sheet_image_draw_ctx = ImageDraw.Draw(sheet_image)
        for i in range(random.randint(0, RANDOM_LINE_COUNT_MAX)):
            line_position = (random.randint(15, 250), random.randint(15, 250))
            line_size = (random.randint(50, 200), random.randint(50, 200))
            sheet_image_draw_ctx.line(
                line_position + line_size, fill=(*random_color(), random.randint(150, 255)), width=1)

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
                        break

                if collision:
                    iterations += 1
                else:
                    fill_color = None
                    if random.random() > 0.5:
                        fill_color = random_color()
                    sheet_image_draw_ctx.ellipse(
                        [ellipse_position, (ellipse_position[0] + ellipse_width,
                                            ellipse_position[1] + ellipse_height)],
                        fill=fill_color, outline=random_color(), width=1
                    )
                    break

        # various image enhancements
        # sharpness
        sharpness_enhancer = PIL.ImageEnhance.Sharpness(sheet_image)
        sheet_image = sharpness_enhancer.enhance(random.uniform(0.2, 2))

        # brightness
        brightness_enhancer = PIL.ImageEnhance.Brightness(sheet_image)
        sheet_image = brightness_enhancer.enhance(random.uniform(0.5, 1.2))

        # contrast
        contrast_enhancer = PIL.ImageEnhance.Contrast(sheet_image)
        sheet_image = contrast_enhancer.enhance(random.uniform(0.5, 1.5))

        # color
        color_enhancer = PIL.ImageEnhance.Color(sheet_image)
        sheet_image = color_enhancer.enhance(random.uniform(0.1, 1.5))

        # rotate 0, 90, 180, or 270 degrees
        random_degree = random.choice([90, 270])
        eq_coords = list(map(lambda coord: BoundingRect.from_coords(
            (coord['x1'], coord['y1']), (coord['x2'], coord['y2'])).rotate((150, 150), random_degree).to_eq_coord(), eq_coords))
        sheet_image = sheet_image.rotate(
            -random_degree, PIL.Image.NEAREST)
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
