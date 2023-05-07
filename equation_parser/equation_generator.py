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
from PIL import Image, ImageDraw, ImageFont, ImageOps
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


CACHE_DIR = './equation_parser/data/images'
TOKENS_FILENAME = 'tokens'
TOKENS_HEADERS = ['eq_id', 'tokens']

FEATURES_FILENAME_PREFIX = 'features'


EQUATION_IMAGE_SIZE = (150, 50)


def to_clean_tokens(rand_numbers):
    return f'{rand_numbers[0]}/{rand_numbers[1]}+{rand_numbers[2]}/{rand_numbers[3]}={rand_numbers[4]}/{rand_numbers[5]}'


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
    return random.randint(2, 4)


def rand_fraction_x_offset():
    return random.randint(3, 8)


def rand_fraction_tilt_offset():
    return random.randint(-2, 2)


def rand_fraction_start_pos():
    return (random.randint(2, 7), random.randint(2, 7))


def rand_denom_y_offset():
    return random.randint(6, 9)


def rand_denom_x_offset():
    return random.randint(0, 5)


def rand_font():
    return f'./assets/fonts-temp/{random.choice(os.listdir("./assets/fonts-temp"))}'


def rand_font_size():
    return random.randint(15, 17)


def rand_rotation_angle():
    return random.randint(-15, 15)
    # return 0


def rand_plus_size():
    return random.randint(16, 19)


def rand_equals_size():
    return random.randint(17, 19)


def rand_operator_x_offset():
    return random.randint(12, 12)


def rand_operator_y_offset():
    return random.randint(10, 10)


def base_font():
    return ImageFont.truetype('./assets/fonts/euler.otf', size=12)


def rand_text_color():
    return (random.randint(215, 255), random.randint(215, 255), random.randint(215, 255))


def rand_background_color():
    return (random.randint(0, 20), random.randint(0, 20), random.randint(0, 20))


def draw_fraction(draw, pos, font, font_size, num, denom):
    # draw.text((pos[0] - 20, pos[1] - 20),
    # num_font_str.split('/')[-1], font=base_font())
    # TODO: rotate each number individually?
    draw.text(pos, str(num), font=font, fill=rand_text_color())
    _, _, num_width, num_height = font.getbbox(str(num))

    denom_pos = (pos[0] + rand_denom_x_offset(), pos[1] +
                 num_height + rand_denom_y_offset())
    draw.text(denom_pos, str(denom), font=font, fill=rand_text_color())

    _, _, denom_width, denom_height = font.getbbox(str(denom))

    line_height = pos[1] + num_height + rand_fraction_y_offset()
    line_width = max(num_width, denom_width)
    line_pos = [pos[0] - rand_fraction_x_offset(), line_height,
                pos[0] + line_width + rand_fraction_x_offset(), line_height + rand_fraction_tilt_offset()]
    draw.line(line_pos, width=rand_fraction_width(), fill=rand_text_color())

    return (pos[0] + line_width, pos[1] + num_height)


def draw_plus(draw, pos):
    plus_font = rand_font()
    font = ImageFont.truetype(plus_font, size=rand_plus_size())
    draw.text(pos, '+', align='top', font=font, fill=rand_text_color())
    return draw.textsize('+', font=font)


def draw_equals(draw, pos):
    equals_font = rand_font()
    font = ImageFont.truetype(equals_font, size=rand_equals_size())
    draw.text(pos, '=', font=font, fill=rand_text_color())
    return draw.textsize('=', font=font)


class EquationGenerator:
    def generate_equation_image(self, cache=True) -> (Image, str):
        eq_tokens = random_equation_tokens()
        background_color = rand_background_color()

        eq_image = Image.new(mode="RGB", size=EQUATION_IMAGE_SIZE,
                             color=background_color)
        font_size = rand_font_size()
        font = ImageFont.truetype(
            rand_font(), size=font_size)
        draw = ImageDraw.Draw(eq_image)
        rand_numbers = [rand_frac_number() for _ in range(6)]
        font_size = rand_font_size()
        fraction_one_start_pos = rand_fraction_start_pos()
        fraction_one = draw_fraction(
            draw, fraction_one_start_pos, font, font_size,
            rand_numbers[0], rand_numbers[1])
        plus_size = draw_plus(
            draw, (fraction_one[0] + rand_operator_x_offset(),
                   fraction_one[1] - rand_operator_y_offset()))
        fraction_two_pos = draw_fraction(
            draw, (fraction_one[0] + plus_size[0] + 17,
                   fraction_one_start_pos[1]), font, font_size,
            rand_numbers[2], rand_numbers[3])

        equals_size = draw_equals(
            draw,
            (fraction_two_pos[0] + rand_operator_x_offset(),
             fraction_two_pos[1] - rand_operator_y_offset()))
        draw_fraction(
            draw,
            (fraction_two_pos[0] + equals_size[0] + 20,
             fraction_one_start_pos[1]), font, font_size,
            rand_numbers[4], rand_numbers[5])

        eq_image = eq_image.rotate(
            rand_rotation_angle(), expand=1, fillcolor=background_color)

        if random.choice([True, False]):
            eq_image = ImageOps.invert(eq_image)

        eq_tokens = to_clean_tokens(rand_numbers)
        if not self.images_cached():
            os.makedirs(CACHE_DIR)
            with open(f'{CACHE_DIR}/{TOKENS_FILENAME}.csv', 'a', newline='', encoding='utf-8') as tokens_file:
                writer = csv.writer(tokens_file)
                writer.writerow(TOKENS_HEADERS)

        cached_eq_id = self.cache_image(eq_image, eq_tokens)
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
