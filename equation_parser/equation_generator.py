import PIL
import random
import sympy
import numpy as np
import sys
import os
import shutil
import csv
import pickle

import uuid
from tensorflow.keras.preprocessing import image

from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib as mpl

from .tokens import START_TOKEN, END_TOKEN


def rand_frac_number():
    return random.randint(1, 999)


def rand_math_font():
    return random.choice([
        'dejavusans',
        'dejavuserif',
        'cm',
        'stix',
        'stixsans'
    ])


def rand_text_color():
    return random.choice([
        'black',
        'midnightblue',
        'indigo',
        'brown',
        'darkred',
        'maroon',
        'blue',
        'red',
        'navy',
    ])


CACHE_DIR = './equation_parser/data'
TOKENS_FILENAME = 'tokens'
TOKENS_HEADERS = ['eq_id', 'tokens']

FEATURES_FILENAME_PREFIX = 'features'


def white_to_transparency(img):
    x = np.asarray(img.convert('RGBA')).copy()
    x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)
    return Image.fromarray(x)


def to_clean_tokens(rand_numbers):
    return f'{rand_numbers[0]} / {rand_numbers[1]} + {rand_numbers[2]} / {rand_numbers[3]} = {rand_numbers[4]} / {rand_numbers[5]}'


def random_equation_tokens():
    rand_numbers = [rand_frac_number() for _ in range(6)]
    return r'\frac{{{a_num}}}{{{a_denom}}}+\frac{{{b_num}}}{{{b_denom}}}=\frac{{{c_num}}}{{{c_denom}}}'.format(
        a_num=rand_numbers[0],
        a_denom=rand_numbers[1],
        b_num=rand_numbers[2],
        b_denom=rand_numbers[3],
        c_num=rand_numbers[4],
        c_denom=rand_numbers[5]
    )


def rand_fraction_width():
    return random.randint(1, 3)


def rand_fraction_y_offset():
    return random.randint(3, 7)


def rand_fraction_x_offset():
    return random.randint(3, 8)


def rand_fraction_tilt_offset():
    return random.randint(-5, 5)


def rand_fraction_start_pos():
    return (random.randint(5, 40), random.randint(5, 100))


def rand_font_size():
    return random.randint(25, 35)


def rand_denom_y_offset():
    return random.randint(3, 7)


def rand_denom_x_offset():
    return random.randint(-4, 4)


def rand_font():
    return f'./assets/fonts/{random.choice(os.listdir("./assets/fonts"))}'


def draw_fraction(draw, pos, font_size, num, denom):
    num_font = ImageFont.truetype(
        rand_font(), size=font_size)
    denom_font = ImageFont.truetype(rand_font(), size=font_size)

    draw.text(pos, str(num), font=num_font)
    num_width, num_height = draw.textsize(str(num), font=num_font)

    line_height = pos[1] + num_height + rand_fraction_y_offset()
    line_pos = [pos[0] - rand_fraction_x_offset(), line_height + rand_fraction_tilt_offset(),
                pos[0] + num_width + rand_fraction_x_offset(), line_height + rand_fraction_tilt_offset()]
    draw.line(line_pos, fill="white", width=rand_fraction_width())

    denom_pos = (pos[0] + rand_denom_x_offset(), pos[1] +
                 num_height + rand_denom_y_offset())
    draw.text(denom_pos, str(denom), font=denom_font)

    return (line_pos[2], line_pos[3])


def draw_plus(draw, pos):
    draw.line([pos[0] + 10, pos[1], pos[0] + 40, pos[1]])
    draw.line([pos[0] + 25, pos[1] - 15, pos[0] + 25, pos[1] + 15])


class EquationGenerator:
    def generate_equation_image(self, dpi=600, cache=True) -> (Image, str):
        eq_tokens = random_equation_tokens()

        eq_image = Image.new(mode="RGBA", size=(300, 200))
        draw = ImageDraw.Draw(eq_image)
        rand_numbers = [rand_frac_number() for _ in range(6)]
        fraction_one_pos = draw_fraction(draw, rand_fraction_start_pos(), rand_font_size(),
                                         rand_numbers[0], rand_numbers[1])
        draw_plus(draw, fraction_one_pos)

        eq_tokens = to_clean_tokens(rand_numbers)
        if not self.images_cached():
            os.makedirs(CACHE_DIR)
            with open(f'{CACHE_DIR}/{TOKENS_FILENAME}.csv', 'a', newline='', encoding='utf-8') as tokens_file:
                writer = csv.writer(tokens_file)
                writer.writerow(TOKENS_HEADERS)

        cached_eq_id = self.cache_image(eq_image, 'eq_tokens')
        return (cached_eq_id, eq_tokens)

    def cache_image(self, eq_image, eq_tokens):
        eq_id = uuid.uuid4()
        with open(f'{CACHE_DIR}/{TOKENS_FILENAME}.csv', 'a', newline='', encoding='utf-8') as tokens_file:
            writer = csv.writer(tokens_file)
            writer.writerow([eq_id, eq_tokens])

        eq_image.save(f'{CACHE_DIR}/{eq_id}.bmp')

        return str(eq_id)

    def images_cached(self):
        return os.path.isdir(CACHE_DIR) and len(
            os.listdir(CACHE_DIR)) != 0

    def equations_from_cache(self):
        if not os.path.isfile(f'{CACHE_DIR}/{TOKENS_FILENAME}.csv'):
            return []

        equations = []

        with open(f'{CACHE_DIR}/{TOKENS_FILENAME}.csv') as tokens_file:
            reader = csv.DictReader(tokens_file)
            for row in reader:
                eq_id = row[TOKENS_HEADERS[0]]
                tokens = row[TOKENS_HEADERS[1]]
                equations.append((eq_id, tokens))

        return equations
