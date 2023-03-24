import PIL
import random
import sympy
import numpy as np
import sys
import os
import shutil

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


def rand_mathtext():
    return random.choice([
        'rm', 'it', 'bf',
        'default', 'regular'
    ])


def rand_text_color():
    return random.choice([
        'black',
        'dimgray',
        'gray',
        'silver',
        'brown',
        'darkred',
        'orange',
        'moccasin',
        'tan',
        'lime',
        'darkgreen',
        'lightsteelblue',
        'blue',
        'green',
        'red',
        'navy',
        'purple'
    ])


def white_to_transparency(img):
    x = np.asarray(img.convert('RGBA')).copy()
    x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)
    return Image.fromarray(x)


class EquationImageGenerator:
    def generate_equation_images(self, image_count, dpi=500, cache_dir=''):
        if len(cache_dir) > 0 and self.images_cached(cache_dir):
            print('Cached equation images found.')
            return self.images_from_cache(cache_dir)

        print('Generating equation images...')
        images = []
        should_cache = len(cache_dir) > 0
        if should_cache and not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)

        for idx in range(image_count):
            im = self.generate_equation_image(dpi)
            if should_cache:
                filename = f'{cache_dir}/eq-{idx}.png'
                im.save(filename)

        if should_cache:
            print('Equation images cached.')

        return images

    def generate_equation_image(self, dpi=500):
        eq_latex = r'\frac{{{a_num}}}{{{a_denom}}}+\frac{{{b_num}}}{{{b_denom}}}=\frac{{{c_num}}}{{{c_denom}}}'.format(
            a_num=rand_frac_number(),
            a_denom=rand_frac_number(),
            b_num=rand_frac_number(),
            b_denom=rand_frac_number(),
            c_num=rand_frac_number(),
            c_denom=rand_frac_number()
        )
        fig = plt.figure()
        mpl.rcParams["mathtext.default"] = rand_mathtext()
        text = fig.text(0, 0, u'${0}$'.format(
            eq_latex), fontsize=2, math_fontfamily=rand_math_font(), color=rand_text_color())
        fig.savefig(BytesIO(), dpi=dpi)
        bbox = text.get_window_extent()
        width, height = bbox.size / float(dpi) + 0.005
        fig.set_size_inches((width, height))

        dy = (bbox.ymin / float(dpi)) / height
        text.set_position((0.01, -dy))

        buffer_ = BytesIO()
        fig.savefig(buffer_, dpi=dpi, transparent=True, format='png')
        plt.close(fig)
        buffer_.seek(0)
        im = Image.open(buffer_)
        return white_to_transparency(im)

    def images_cached(self, cache_dir):
        return os.path.isdir(cache_dir) and len(
            os.listdir(cache_dir)) != 0

    def images_from_cache(self, cache_dir):
        images = []

        for filename in os.listdir(cache_dir):
            image_file = os.path.join(cache_dir, filename)
            if os.path.isfile(image_file):
                equation_image = PIL.Image.open(image_file)
                images.append(equation_image)

        return images
