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


def to_padded_tokens(rand_numbers):
    return f'{rand_numbers[0]}/{rand_numbers[1]}+{rand_numbers[2]}/{rand_numbers[3]}={rand_numbers[4]}/{rand_numbers[5]}'


class EquationImageGenerator:
    def __init__(self, feature_extractor_model):
        self.feature_extractor_model = feature_extractor_model

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
        eq_tokens = to_padded_tokens(rand_numbers)
        if not self.images_cached():
            os.makedirs(CACHE_DIR)
            with open(f'{CACHE_DIR}/{TOKENS_FILENAME}.csv', 'a') as tokens_file:
                writer = csv.writer(tokens_file)
                writer.writerow(TOKENS_HEADERS)

        cached_eq_id = self.cache_image(eq_image, eq_tokens)
        features = self.cache_image_features(eq_image, cached_eq_id)
        return (eq_image, eq_tokens, features, cached_eq_id)

    def cache_image(self, eq_image, eq_tokens):
        eq_id = uuid.uuid4()
        with open(f'{CACHE_DIR}/{TOKENS_FILENAME}.csv', 'a') as tokens_file:
            writer = csv.writer(tokens_file)
            writer.writerow([eq_id, eq_tokens])

        eq_image.save(f'{CACHE_DIR}/{eq_id}.bmp')

        return eq_id

    def cache_image_features(self, eq_image, uuid):
        features_filename = f'{CACHE_DIR}/{FEATURES_FILENAME_PREFIX}-{uuid}.p'
        if os.path.isfile(features_filename):
            return

        eq_image = eq_image.resize(
            (100, 100), resample=PIL.Image.BILINEAR)
        eq_image = eq_image.convert('RGB')
        eq_image_data = image.img_to_array(eq_image)
        image_to_predict = np.expand_dims(eq_image_data, axis=0)

        features = self.feature_extractor_model.predict(image_to_predict)
        features = np.array(features[0]).astype('float32')
        pickle.dump(features, open(features_filename, 'wb'))

        return features

    def features_from_equation_id(self, eq_id):
        feature_filename = f'{CACHE_DIR}/{FEATURES_FILENAME_PREFIX}-{eq_id}.p'
        return pickle.load(open(feature_filename, 'rb'))

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

                if os.path.isfile(f'{CACHE_DIR}/{eq_id}.bmp'):
                    eq_image = PIL.Image.open(f'{CACHE_DIR}/{eq_id}.bmp')
                else:
                    eq_image = None

                features = self.features_from_equation_id(eq_id)

                equations.append((eq_image, tokens, features, eq_id))

        return equations
