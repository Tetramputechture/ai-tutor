from equation_analyzer.equation_image_utils.equation_image_utils import equation_image
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from .equation_box import EquationBox
from skimage.util import random_noise
import numpy as np
import string
import random
import sys
import os


def random_text():
    text_len = random.randint(5, 15)
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + (string.digits + '+-=') * 4)
                   for _ in range(text_len))


def random_color():
    return tuple(random.choices(range(256), k=3))


def random_sheet_color():
    return random.choice([
        'aliceblue',
        'whitesmoke',
        'white',
        'lightgray',
        'linen',
        'lavender',
        'snow',
        'bisque'
    ])


def rand_frac_number():
    return random.randint(1, 999)


def rand_scale_factor():
    return random.uniform(0.4, 0.5)


def rand_eq_sheet_padding():
    return random.randint(5, 20)


FONTS_FOLDER = './assets/fonts'


def rand_font():
    return f'{FONTS_FOLDER}/{random.choice(os.listdir(FONTS_FOLDER))}'


class EquationSheetDecorator:
    def add_equation(sheet_image, eq_boxes=[], include_sheet_bg=False):
        sheet_width, sheet_height = sheet_image.size

        rand_numbers = [rand_frac_number() for _ in range(6)]
        # equation_image, _ = EquationImageGenerator().generate_equation_image()
        eq_image = equation_image(rand_numbers, False)
        original_image_width, original_image_height = eq_image.size

        scale_factor = rand_scale_factor()
        eq_image = eq_image.resize(
            (int(original_image_width * scale_factor), int(original_image_height * scale_factor)), Image.BICUBIC)

        # eq_image = EquationSheetDecorator.adjust_brightness(eq_image, 2)

        image_width, image_height = eq_image.size

        max_x_pos, max_y_pos = (
            (sheet_width - image_width - 2),
            (sheet_height - image_height - 5)
        )

        eq_position = (random.randint(
            1, max_x_pos), random.randint(0, max_y_pos))

        eq_box = EquationBox(
            eq_position, (eq_position[0] + image_width, eq_position[1] + image_height))

        if include_sheet_bg:
            sheet_bg_rect = [
                eq_box.topLeft[0] - rand_eq_sheet_padding(),
                eq_box.topLeft[1] - rand_eq_sheet_padding(),
                eq_box.bottomRight[0] + rand_eq_sheet_padding(),
                eq_box.bottomRight[1] + rand_eq_sheet_padding()
            ]
            ImageDraw.Draw(sheet_image).rectangle(sheet_bg_rect, fill='white')

        sheet_image.paste(
            eq_image, (int(eq_position[0]), int(eq_position[1])), eq_image)

        eq_box = EquationBox((eq_position[0], eq_position[1]),
                             (eq_position[0] + image_width, eq_position[1] + image_height))

        return eq_box

    def adjust_sharpness(sheet_image, value=1):
        return ImageEnhance.Sharpness(sheet_image).enhance(value)

    def adjust_brightness(sheet_image, value=1):
        return ImageEnhance.Brightness(sheet_image).enhance(value)

    def adjust_contrast(sheet_image, value=1):
        return ImageEnhance.Contrast(sheet_image).enhance(value)

    def adjust_color(sheet_image, value=1):
        return ImageEnhance.Color(sheet_image).enhance(value)

    def invert_color(sheet_image):
        return ImageOps.invert(sheet_image.convert('RGB'))

    def add_noise(sheet_image, extra=False):
        im_arr = np.asarray(sheet_image)
        rand_variance = random.uniform(0.001, 0.002)
        if extra:
            rand_variance = random.uniform(0.01, 0.05)
        noise_img = random_noise(im_arr, mode='gaussian', var=rand_variance)
        noise_img = (255*noise_img).astype(np.uint8)
        return Image.fromarray(noise_img)

    def rotate_sheet(sheet_image, eq_boxes, rotation_degrees):
        sheet_size = sheet_image.size
        sheet_center = (
            int(sheet_image.size[0] / 2), int(sheet_image.size[1] / 2))
        new_sheet_image = sheet_image.rotate(
            -rotation_degrees, Image.BICUBIC)
        new_eq_boxes = list(map(lambda box: box.rotate(
            sheet_center, rotation_degrees), eq_boxes))

        return (new_sheet_image, new_eq_boxes)
