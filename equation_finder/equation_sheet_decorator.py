import string
import random
import sys

from .equation_box import EquationBox
from .equation_image_generator import EquationImageGenerator

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


def random_text():
    text_len = random.randint(3, 12)
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
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


def random_font():
    return random.choice([
        './assets/fonts/ArefRuqaa-Regular.ttf',
        './assets/fonts/BungeeColor-Regular_colr_Windows.ttf',
        './assets/fonts/FreeMono.ttf',
        './assets/fonts/NotoSans-Regular.ttf',
        './assets/fonts/OpenSansCondensed-LightItalic.ttf'
    ])


class EquationSheetDecorator:
    def add_text(sheet_image, text_count, eq_boxes):
        text_image = Image.new(
            'RGBA', sheet_image.size, (255, 255, 255, 0))
        image_draw_ctx = ImageDraw.Draw(text_image)
        sheet_size = sheet_image.size

        # add misc other text
        for i in range(text_count):
            iterations = 0

            fnt = ImageFont.truetype(random_font(), random.randint(6, 14))
            text = random_text()
            while iterations < 1000000:
                text_position = (random.randint(1, sheet_size[0]),
                                 random.randint(1, sheet_size[1]))
                image_draw_ctx.text(text_position, text,
                                    font=fnt, fill=(*random_color(), 0))

                text_width, text_height = image_draw_ctx.textsize(
                    text, font=fnt, spacing=4)

                text_bounding_rect = EquationBox(
                    text_position,
                    (text_position[0] + text_width,
                     text_position[1] + text_height)
                )

                collision = False
                for box in eq_boxes:
                    if box.collision(text_bounding_rect):
                        collision = True
                        break

                if collision:
                    iterations += 1
                else:
                    image_draw_ctx.text(text_position, text,
                                        font=fnt, fill=(*random_color(), random.randint(150, 255)))

                    sheet_image = Image.alpha_composite(
                        sheet_image, text_image)
                    break

        return sheet_image

    def add_lines(sheet_image, line_count):
        sheet_image_draw_ctx = ImageDraw.Draw(sheet_image)
        sheet_size = sheet_image.size
        max_x_pos, max_y_pos = (
            (sheet_size[0] - 20),
            (sheet_size[1] - 20)
        )

        for i in range(line_count):
            line_position = (random.randint(1, max_x_pos),
                             random.randint(1, max_y_pos))
            line_size = (random.randint(
                20, max_x_pos - line_position[0] + 20), random.randint(20, max_y_pos - line_position[1] + 20))
            sheet_image_draw_ctx.line(
                line_position + line_size, fill=(*random_color(), random.randint(150, 255)), width=1)

        return sheet_image

    def add_ellipses(sheet_image, ellipse_count, eq_boxes):
        sheet_image_draw_ctx = ImageDraw.Draw(sheet_image)
        sheet_size = sheet_image.size
        max_x_pos, max_y_pos = (
            (sheet_size[0]),
            (sheet_size[1])
        )

        for i in range(ellipse_count):
            iterations = 0
            while iterations < 1000000:
                ellipse_position = (random.randint(1, max_x_pos),
                                    random.randint(1, max_y_pos))
                ellipse_width, ellipse_height = (
                    random.randint(5, 30), random.randint(5, 30))

                ellipse_bounding_rect = EquationBox(
                    ellipse_position,
                    (ellipse_position[0] + ellipse_width,
                     ellipse_position[1] + ellipse_height)
                )

                collision = False

                for box in eq_boxes:
                    if box.collision(ellipse_bounding_rect):
                        collision = True
                        break

                if collision:
                    iterations += 1
                else:
                    fill_color = None
                    if random.random() > 0.5:
                        fill_color = random_color()
                    sheet_image_draw_ctx.ellipse(
                        [ellipse_position, (ellipse_position[0] + ellipse_width,
                                            ellipse_position[1] + ellipse_height)],
                        fill=fill_color, outline=random_color(), width=1
                    )
                    break

        return sheet_image

    def add_equation(sheet_image, eq_boxes=[]):
        sheet_image_draw_ctx = ImageDraw.Draw(sheet_image)
        sheet_width, sheet_height = sheet_image.size

        equation_image = EquationImageGenerator().generate_equation_image()
        original_image_width, original_image_height = equation_image.size

        iterations = 0
        while iterations < 1000000:
            scale_factor = random.uniform(0.15, 0.25)

            equation_image = equation_image.resize(
                (int(original_image_width * scale_factor), int(original_image_height * scale_factor)), Image.BICUBIC)

            image_width, image_height = equation_image.size

            max_x_pos, max_y_pos = (
                (sheet_width - image_width - 15),
                (sheet_height - image_height - 30)
            )

            eq_position = (random.randint(
                15, max_x_pos), random.randint(15, max_y_pos))

            eq_box = EquationBox(
                eq_position, (eq_position[0] + image_width, eq_position[1] + image_height))

            collision = False

            sys.stdout.flush()

            for box in eq_boxes:
                if box.collision(eq_box):
                    collision = True
                    break

            if collision:
                iterations += 1
            else:
                sheet_image.paste(
                    equation_image, (int(eq_position[0]), int(eq_position[1])), equation_image)
                break

        return EquationBox(
            eq_position,
            (eq_position[0] + image_width,
                eq_position[1] + image_height)
        )

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

    def rotate_sheet(sheet_image, eq_boxes, rotation_degrees):
        sheet_size = sheet_image.size
        sheet_center = (
            int(sheet_image.size[0] / 2), int(sheet_image.size[1] / 2))
        new_sheet_image = sheet_image.rotate(
            -rotation_degrees, Image.BICUBIC)
        new_eq_boxes = list(map(lambda box: box.rotate(
            sheet_center, rotation_degrees), eq_boxes))

        return (new_sheet_image, new_eq_boxes)
