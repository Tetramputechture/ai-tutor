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
from PIL import Image
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


class EquationGenerator:
    def generate_equation_image(self, dpi=600, cache=True) -> (Image, str):
        rand_numbers = [rand_frac_number() for _ in range(6)]
        eq_latex = r'\frac{{{a_num}}}{{{a_denom}}}+\frac{{{b_num}}}{{{b_denom}}}=\frac{{{c_num}}}{{{c_denom}}}'.format(
            a_num=rand_numbers[0],
            a_denom=rand_numbers[1],
            b_num=rand_numbers[2],
            b_denom=rand_numbers[3],
            c_num=rand_numbers[4],
            c_denom=rand_numbers[5]
        )
        fig = plt.figure()
        rotation_degrees = random.randint(-15, 15)
        text = fig.text(0, 0, u'${0}$'.format(
            eq_latex), fontsize=6, math_fontfamily=rand_math_font(), color=rand_text_color(), rotation=rotation_degrees, rotation_mode="anchor")
        fig.savefig(BytesIO(), dpi=dpi)
        bbox = text.get_window_extent()
        width, height = bbox.size / float(dpi)
        fig.set_size_inches((width, height))

        dy = (bbox.ymin / float(dpi)) / height
        dx = (bbox.xmin / float(dpi)) / width
        text.set_position((-dx, -dy))

        buffer_ = BytesIO()
        fig.savefig(buffer_, dpi=dpi, transparent=True, format='png')
        plt.close(fig)
        buffer_.seek(0)
        im = Image.open(buffer_)
        eq_image = white_to_transparency(im)
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
