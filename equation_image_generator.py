import PIL
import random
import sympy
import numpy as np
import sys

from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt


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
    def generate_equation_images(self, image_count, dpi=400):
        images = []
        for i in range(image_count):
            eq_latex = r'\frac{{{a_num}}}{{{a_denom}}}+\frac{{{b_num}}}{{{b_denom}}}=\frac{{{c_num}}}{{{c_denom}}}'.format(
                a_num=rand_frac_number(),
                a_denom=rand_frac_number(),
                b_num=rand_frac_number(),
                b_denom=rand_frac_number(),
                c_num=rand_frac_number(),
                c_denom=rand_frac_number()
            )
            fig = plt.figure()
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
            im = white_to_transparency(im)
            images.append(im)
            print('.', end='')

        return images
