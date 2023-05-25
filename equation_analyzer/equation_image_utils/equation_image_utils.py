import random
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.util import random_noise

EQUATION_WIDTH_PX = 400

EQUATION_IMAGE_SIZE = (EQUATION_WIDTH_PX, int(EQUATION_WIDTH_PX * 0.25))

FONTS_FOLDER = './assets/fonts-temp'


def white_to_transparent(img):
    width, height = img.size
    pixdata = img.load()
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (255, 255, 255, 255):
                pixdata[x, y] = (255, 255, 255, 0)

    return img


def equation_image(numbers, background=True) -> (Image, str):
    if background:
        background_color = rand_background_color()
    else:
        background_color = (0, 0, 0)

    eq_image = Image.new(mode="RGB", size=EQUATION_IMAGE_SIZE,
                         color=background_color)
    font_size = rand_font_size()
    font = ImageFont.truetype(
        rand_font(), size=font_size)
    draw = ImageDraw.Draw(eq_image)
    font_size = rand_font_size()
    fraction_one_start_pos = rand_fraction_start_pos()
    fraction_one = draw_fraction(
        draw, fraction_one_start_pos, font, font_size,
        numbers[0], numbers[1])
    plus_size = draw_plus(
        draw, (fraction_one[0] + rand_operator_x_offset(),
               fraction_one[1] - rand_operator_y_offset()))
    fraction_two_pos = draw_fraction(
        draw, (fraction_one[0] + plus_size[0] + rand_operator_post_x_offset(),
               fraction_one_start_pos[1]), font, font_size,
        numbers[2], numbers[3])

    equals_size = draw_equals(
        draw,
        (fraction_two_pos[0] + rand_operator_x_offset(),
         fraction_two_pos[1] - rand_operator_y_offset()))
    draw_fraction(
        draw,
        (fraction_two_pos[0] + equals_size[0] + rand_operator_post_x_offset(),
         fraction_one_start_pos[1]), font, font_size,
        numbers[4], numbers[5])

    eq_image = eq_image.rotate(
        rand_rotation_angle(), expand=1, fillcolor=background_color)

    if background:
        eq_image = draw_noise(eq_image)
        if random.choice([True, False]):
            eq_image = ImageOps.invert(eq_image)
    else:
        eq_image = ImageOps.invert(eq_image)
        eq_image = eq_image.convert('RGBA')
        eq_image = white_to_transparent(eq_image)

    return eq_image


def rand_fraction_width():
    return random.randint(4, 5)


def rand_fraction_y_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.015)
    rand_begin = int(EQUATION_WIDTH_PX * 0.015)
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_fraction_x_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.03)
    rand_begin = int(EQUATION_WIDTH_PX * 0.01)
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_fraction_tilt_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.02)
    rand_begin = int(EQUATION_WIDTH_PX * 0.01)
    return random.randint(-rand_begin, rand_begin + rand_range)


def rand_fraction_start_pos():
    rand_range_x = int(EQUATION_WIDTH_PX * 0.03)  # 5
    rand_begin_x = int(EQUATION_WIDTH_PX * 0.01)  # 2
    rand_range_y = int(EQUATION_WIDTH_PX * 0.03)  # 5
    rand_begin_y = int(EQUATION_WIDTH_PX * 0.005)  # 2
    x_coord = random.randint(rand_begin_x, rand_begin_x + rand_range_x)
    y_coord = random.randint(rand_begin_y, rand_begin_y + rand_range_x)
    return (x_coord, y_coord)


def rand_denom_y_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.02)  # 3
    rand_begin = int(EQUATION_WIDTH_PX * 0.05)  # 6
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_denom_x_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.035)  # 5
    rand_begin = 0
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_font():
    return f'{FONTS_FOLDER}/{random.choice(os.listdir(FONTS_FOLDER))}'


def rand_font_size():
    rand_begin = int(EQUATION_WIDTH_PX * 0.07)  # 15
    return rand_begin


def rand_rotation_angle():
    return random.randint(-10, 10)


def rand_plus_size():
    rand_range = int(EQUATION_WIDTH_PX * 0.02)  # 3
    rand_begin = int(EQUATION_WIDTH_PX * 0.13)  # 16
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_equals_size():
    rand_range = int(EQUATION_WIDTH_PX * 0.015)  # 2
    rand_begin = int(EQUATION_WIDTH_PX * 0.125)  # 17
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_operator_x_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.01)  # 2
    rand_begin = int(EQUATION_WIDTH_PX * 0.07)  # 12
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_operator_y_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.015)  # 2
    rand_begin = int(EQUATION_WIDTH_PX * 0.055)  # 10
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_operator_post_x_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.015)  # 2
    rand_begin = int(EQUATION_WIDTH_PX * 0.13)  # 17
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_text_color():
    return (random.randint(235, 255), random.randint(235, 255), random.randint(235, 255))


def rand_background_color():
    return (random.randint(0, 15), random.randint(0, 15), random.randint(0, 15))


def draw_fraction(draw, pos, font, font_size, num, denom):
    # TODO: rotate each number individually?
    draw.text(pos, str(num), font=font,
              fill=rand_text_color(), spacing=5)
    _, _, num_width, num_height = font.getbbox(str(num))

    denom_pos = (pos[0] + rand_denom_x_offset(), pos[1] +
                 num_height + rand_denom_y_offset())
    draw.text(denom_pos, str(denom), font=font,
              fill=rand_text_color(), spacing=5)

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


def draw_noise(eq_image):
    im_arr = np.asarray(eq_image)
    rand_variance = random.uniform(0.001, 0.003)
    noise_img = random_noise(im_arr, mode='gaussian', var=rand_variance)
    noise_img = (255*noise_img).astype(np.uint8)
    return Image.fromarray(noise_img)
