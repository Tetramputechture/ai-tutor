import random
import os
import numpy as np
import cv2
import csv
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.util import random_noise
import albumentations as A

EQUATION_WIDTH_PX = 430

EQUATION_IMAGE_SIZE = (EQUATION_WIDTH_PX, int(EQUATION_WIDTH_PX * 0.25))

FONTS_FOLDER = './assets/fonts-temp'
MNIST_FOLDER = './data/mnist_numbers'
EQ_FOLDER = './data/equations_custom'
PLUS_FOLDER = './data/mnist_numbers/add'
EQL_FOLDER = './data/mnist_numbers/eq'


def rand_mnist_number(num):
    full_folder_name = f'{MNIST_FOLDER}/{num}'
    return Image.open(f'{full_folder_name}/{random.choice(os.listdir(full_folder_name))}')


def white_to_transparent(img):
    width, height = img.size
    pixdata = img.load()
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (255, 255, 255, 255):
                pixdata[x, y] = (255, 255, 255, 0)

    return img


def augment_img(img):
    cv_img = np.array(img)

    # if random.randint(1, 5) == 1:
    #     img = erode_image(cv_img)
    if random.randint(1, 3) == 1:
        cv_img = dilate_image(cv_img)

    transform = A.Compose([
        A.OneOf([
            # add black pixels noise
            A.OneOf([
                # A.RandomRain(brightness_coefficient=1, drop_length=90, drop_width=1, drop_color=(
                #     0, 0, 0), blur_value=1, rain_type='drizzle', p=0.05),
                A.RandomShadow(shadow_dimension=4, num_shadows_upper=3, p=1),
                # A.RandomSnow(brightness_coeff=1.5, always_apply=True,  p=1),
                A.Spatter(intensity=0.35, p=1),
                # A.RandomRain(
                #     slant_lower=-20,
                #     slant_upper=-16,
                #     brightness_coefficient=1,
                #     drop_length=1,
                #     drop_width=1,
                #     drop_color=(40, 40, 40),
                #     blur_value=1,
                #     rain_type='drizzle',
                #     p=0.5
                # ),
            ], p=1),

            # add white pixels noise
            A.OneOf([
                # A.PixelDropout(drop_value=255, p=1),
                A.RandomShadow(shadow_dimension=3, num_shadows_upper=4, p=1),
                A.RandomSnow(brightness_coeff=2, always_apply=True, p=1),
                A.Spatter(intensity=0.35, p=1),
                # A.RandomRain(
                #     slant_lower=-20,
                #     slant_upper=-16,
                #     brightness_coefficient=1,
                #     drop_length=1,
                #     drop_width=1,
                #     drop_color=(40, 40, 40),
                #     blur_value=1,
                #     rain_type='drizzle',
                #     p=0.5
                # ),
            ], p=1),
        ], p=1),

        # transformations
        A.ShiftScaleRotate(
            shift_limit=0.005,
            scale_limit=[-0.05, 0.01],
            rotate_limit=4,
            value=(255, 255, 255),
            border_mode=cv2.BORDER_CONSTANT,
            p=1),
        A.Blur(blur_limit=3, p=0.6),
    ])

    cv_img = transform(image=cv_img)['image']
    return Image.fromarray(cv_img)


def dilate_image(cv_img):
    rand_dilation_amount = random.randint(1, 3)
    # if rand_dilation_amount == 0:
    #     return img

    kernel = np.ones((rand_dilation_amount, rand_dilation_amount), np.uint8)
    cv_img = cv2.dilate(cv_img, kernel)
    return cv_img
    # return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def erode_image(cv_img):
    rand_erosion_amount = random.randint(0, 1)
    if rand_erosion_amount == 0:
        return cv_img

    kernel = np.ones((rand_erosion_amount, rand_erosion_amount), np.uint8)
    cv_img = cv2.erode(cv_img, kernel)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


CUSTOM_EQUATIONS = []

with open('./data/equations_custom/tokens.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for line in reader:
        CUSTOM_EQUATIONS.append(line)

EQ_JPG_FILES = [f for f in os.listdir(
    EQ_FOLDER) if f.endswith('.jpg')]


def custom_equation_image():
    filename = random.choice(EQ_JPG_FILES)
    eq_id = filename.split('.')[0]
    tokens = [eq for eq in CUSTOM_EQUATIONS if eq['eq_id'] == eq_id]
    image = Image.open(f'{EQ_FOLDER}/{filename}').resize(EQUATION_IMAGE_SIZE)
    # if random.choice([True, False]):
    #     image = ImageOps.invert(image)
    return (image, tokens[0]['tokens'])


def equation_image(numbers, background=True) -> (Image, str):
    if background:
        background_color = rand_background_color()
    else:
        background_color = (0, 0, 0)

    eq_image = Image.new(mode="RGB", size=EQUATION_IMAGE_SIZE,
                         color=background_color)
    draw = ImageDraw.Draw(eq_image)
    fraction_one_start_pos = rand_fraction_start_pos()
    fraction_one = draw_fraction_mnist(
        eq_image, draw, fraction_one_start_pos,
        numbers[0], numbers[1])
    plus_size = draw_plus(
        draw, (fraction_one[0] + rand_operator_x_offset(),
               fraction_one[1] - rand_operator_y_offset()))
    fraction_two_pos = draw_fraction_mnist(
        eq_image, draw, (fraction_one[0] + plus_size[0] + rand_operator_post_x_offset(),
                         fraction_one_start_pos[1]),
        numbers[2], numbers[3])

    equals_size = draw_equals(
        draw,
        (fraction_two_pos[0] + rand_operator_x_offset(),
            fraction_two_pos[1] - rand_operator_y_offset()))
    eq_x, _, = draw_fraction_mnist(
        eq_image,
        draw,
        (fraction_two_pos[0] + equals_size[0] + rand_operator_post_x_offset(),
            fraction_one_start_pos[1]),
        numbers[4], numbers[5])

    # # crop eq image to end of equation
    eq_image = eq_image.crop(
        (0, 0, min(eq_x + 10, eq_image.size[0]), eq_image.size[1]))

    # dilate or erode
    eq_image = augment_img(eq_image)
    # else:
    #     eq_image = erode_image(eq_image)

    # rotate
    # eq_image = eq_image.rotate(
    #     rand_rotation_angle(), expand=1, fillcolor=background_color)

    if background:
        if random.choice([True, False]):
            eq_image = ImageOps.invert(eq_image)
    if not background:
        # eq_image = draw_noise(eq_image)
        eq_image = ImageOps.invert(eq_image)
        eq_image = eq_image.convert('RGBA')
        eq_image = white_to_transparent(eq_image)

    return eq_image


def rand_fraction_width():
    return random.randint(3, 5)


def rand_text_spacing():
    return 25


def rand_fraction_y_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.015)
    rand_begin = int(EQUATION_WIDTH_PX * 0.005)
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_fraction_x_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.03)
    rand_begin = int(EQUATION_WIDTH_PX * 0.01)
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_fraction_tilt_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.015)
    rand_begin = int(EQUATION_WIDTH_PX * 0.005)
    return random.randint(-rand_begin, rand_begin + rand_range)


def rand_fraction_start_pos():
    rand_range_x = int(EQUATION_WIDTH_PX * 0.03)  # 5
    rand_begin_x = int(EQUATION_WIDTH_PX * 0.015)  # 2
    rand_range_y = int(EQUATION_WIDTH_PX * 0.02)  # 5
    rand_begin_y = int(EQUATION_WIDTH_PX * 0.01)   # 2
    x_coord = random.randint(rand_begin_x, rand_begin_x + rand_range_x)
    y_coord = random.randint(rand_begin_y, rand_begin_y + rand_range_y)
    return (x_coord, y_coord)


def rand_denom_y_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.005)  # 3
    rand_begin = int(EQUATION_WIDTH_PX * 0.045)  # 6
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_denom_x_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.025)  # 5
    rand_begin = 0
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_font():
    return f'{FONTS_FOLDER}/{random.choice(os.listdir(FONTS_FOLDER))}'


def rand_font_size():
    return int(EQUATION_WIDTH_PX * 0.085)
    # return random.randint(
    #     int(EQUATION_WIDTH_PX * 0.08), int(EQUATION_WIDTH_PX * 0.08))  # 15


def rand_rotation_angle():
    # return 0
    return random.randint(-5, 5)


def rand_plus_size():
    rand_range = int(EQUATION_WIDTH_PX * 0.05)  # 3
    rand_begin = int(EQUATION_WIDTH_PX * 0.12)  # 16
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_equals_size():
    rand_range = int(EQUATION_WIDTH_PX * 0.03)  # 2
    rand_begin = int(EQUATION_WIDTH_PX * 0.1)  # 17
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_operator_x_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.015)  # 2
    rand_begin = int(EQUATION_WIDTH_PX * 0.01)  # 12
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_operator_y_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.02)  # 2
    rand_begin = int(EQUATION_WIDTH_PX * 0.07)  # 10
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_operator_post_x_offset():
    rand_range = int(EQUATION_WIDTH_PX * 0.015)  # 2
    rand_begin = int(EQUATION_WIDTH_PX * 0.06)  # 17
    return random.randint(rand_begin, rand_begin + rand_range)


def rand_text_color():
    return (random.randint(235, 255), random.randint(235, 255), random.randint(235, 255))


def rand_background_color():
    return (random.randint(0, 15), random.randint(0, 15), random.randint(0, 15))


def rand_number_spacing():
    return random.randint(-8, 0)


def rand_number_size():
    return (random.randint(26, 29), random.randint(33, 38))


def draw_fraction_mnist(img, draw, pos, num, denom):
    # draw each number in numerator
    number_pos_x = pos[0]
    max_number_height = 0
    num_width = 0
    for number in str(num):
        number_pos = (number_pos_x, pos[1])
        number_size = rand_number_size()
        mnist_img = rand_mnist_number(number)
        mnist_img = mnist_img.resize(number_size)
        img.paste(mnist_img, number_pos, mnist_img)
        number_pos_x += number_size[0] + rand_number_spacing()
        num_width += number_size[0]
        max_number_height = max(max_number_height, number_size[1])

    # draw each number in denominator
    number_pos_x = pos[0]
    denom_width = 0
    for number in str(denom):
        number_pos = (number_pos_x, max_number_height + rand_denom_y_offset())
        number_size = rand_number_size()
        mnist_img = rand_mnist_number(number)
        mnist_img = mnist_img.resize(number_size)
        img.paste(mnist_img, number_pos, mnist_img)
        number_pos_x += number_size[0]
        denom_width += number_size[0]

    line_height = pos[1] + max_number_height + rand_fraction_y_offset()
    line_width = max(num_width, denom_width)
    line_pos = [pos[0] - rand_fraction_x_offset(), line_height,
                pos[0] + line_width + rand_fraction_x_offset(), line_height + rand_fraction_tilt_offset()]
    draw.line(line_pos, width=rand_fraction_width(), fill=rand_text_color())

    return (pos[0] + (line_pos[2] - pos[0]), pos[1] + max_number_height)


def rand_plus_img():
    return Image.open(f'{PLUS_FOLDER}/{random.choice(os.listdir(PLUS_FOLDER))}')


def rand_eql_img():
    return Image.open(f'{EQL_FOLDER}/{random.choice(os.listdir(EQL_FOLDER))}')


def rand_operator_size():
    return random.randint(40, 50)


def draw_plus(draw, pos):
    #     plus_img = ImageOps.invert(rand_plus_img())
    #     new_size = (rand_operator_size(), rand_operator_size())
    #     plus_img = plus_img.resize(new_size)
    #     img.paste(plus_img, pos, plus_img)
    #     return new_size
    plus_font = rand_font()
    font = ImageFont.truetype(plus_font, size=rand_plus_size())
    draw.text(pos, '+', align='top', font=font, fill=rand_text_color())
    return draw.textsize('+', font=font)


def draw_equals(draw, pos):
    # eql_img = ImageOps.invert(rand_eql_img().convert('RGB'))
    # new_size = (rand_operator_size(), rand_operator_size())
    # eql_img = eql_img.resize(new_size)
    # img.paste(eql_img, pos)
    # return new_size
    eql_font = rand_font()
    font = ImageFont.truetype(eql_font, size=rand_equals_size())
    draw.text(pos, '=', align='top', font=font, fill=rand_text_color())
    return draw.textsize('=', font=font)


def draw_noise(eq_image):
    im_arr = np.asarray(eq_image)
    rand_variance = random.uniform(0.002, 0.015)
    noise_img = random_noise(im_arr, mode='gaussian', var=rand_variance)
    noise_img = (255*noise_img).astype(np.uint8)
    return Image.fromarray(noise_img)
