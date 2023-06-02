import PIL
import random
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

from ..equation_image_utils.equation_image_utils import augment_img, equation_image, custom_equation_image

TOKENS_FILENAME = 'tokens'
TOKENS_HEADERS = ['eq_id', 'tokens']


def to_clean_tokens(rand_numbers):
    return f'{rand_numbers[0]}/{rand_numbers[1]}+{rand_numbers[2]}/{rand_numbers[3]}={rand_numbers[4]}/{rand_numbers[5]}'


def rand_frac_number():
    return random.randint(1, 999)


class EquationGenerator:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def generate_equation_image(self, cache=True) -> (Image, str):
        rand_numbers = [rand_frac_number() for _ in range(6)]
        if random.randint(1, 2) == 1:
            eq_image, eq_tokens = custom_equation_image()
            eq_image = augment_img(eq_image)
        else:
            eq_image = equation_image(rand_numbers)
            eq_tokens = to_clean_tokens(rand_numbers)

        # if not self.images_cached():
        #     os.makedirs(self.cache_dir)
        #     with open(f'{self.cache_dir}/{TOKENS_FILENAME}.csv', 'a', newline='', encoding='utf-8') as tokens_file:
        #         writer = csv.writer(tokens_file)
        #         writer.writerow(TOKENS_HEADERS)

        cached_eq_id = self.cache_image(eq_image, eq_tokens)
        return (cached_eq_id, eq_tokens)

    def cache_image(self, eq_image, eq_tokens):
        eq_id = uuid.uuid4()
        # with open(f'{self.cache_dir}/{TOKENS_FILENAME}.csv', 'a', newline='', encoding='utf-8') as tokens_file:
        #     writer = csv.writer(tokens_file)
        #     writer.writerow([eq_id, eq_tokens])

        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        eq_image.save(f'{self.cache_dir}/{eq_id}.bmp')

        return str(eq_id)

    def images_cached(self):
        return os.path.isdir(self.cache_dir) and len(
            os.listdir(self.cache_dir)) != 0

    def equations_from_cache(self):
        if not os.path.isfile(f'{self.cache_dir}/{TOKENS_FILENAME}.csv'):
            return []

        equations = []

        with open(f'{self.cache_dir}/{TOKENS_FILENAME}.csv') as tokens_file:
            reader = csv.DictReader(tokens_file)
            for row in reader:
                eq_id = row[TOKENS_HEADERS[0]]
                tokens = row[TOKENS_HEADERS[1]]
                equations.append((eq_id, tokens))

        return equations
