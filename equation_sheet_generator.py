import random
from pdf2image import convert_from_path
import PIL
import math
import json

from bounding_rect import BoundingRect
from equation_image_generator import EquationImageGenerator

MIN_EQUATION_WIDTH = 40
MIN_EQUATION_HEIGHT = 10

MAX_EQUATION_WIDTH = 150
MAX_EQUATION_HEIGHT = 50


class EquationSheetGenerator:
    def __init__(self, equation_images, max_equations_per_sheet):
        self.equation_images = equation_images
        self.max_equations_per_sheet = max_equations_per_sheet

    def generate_sheet_images(self, sheet_count):
        sheet_images = []
        for i in range(sheet_count):
            sheet_images.append(self.generate_sheet_image())
            print('.', end='')
        return sheet_images

    def generate_sheet_image(self):
        bounding_rects = []
        eq_coords = []
        sheet_image = PIL.Image.new(
            mode="RGBA", size=(300, 300), color=(tuple(random.choices(range(256), k=3))))
        num_equations = random.randint(1, self.max_equations_per_sheet)
        for equation_image in random.sample(self.equation_images, num_equations):
            # for each equation image, choose random location on sheet
            eq_width, eq_height = equation_image.size

            iterations = 0
            while iterations < 1000000:
                eq_position = (random.randint(1, 300), random.randint(1, 300))
                eq_bounding_rect = BoundingRect.from_coords(
                    (eq_position[0], eq_position[1]),
                    (eq_position[0] + eq_width, eq_position[1] + eq_height))
                scale_factor = random.uniform(-0.25, 0.4) + 1
                eq_bounding_rect = eq_bounding_rect.scale(scale_factor)

                collision = False
                if eq_position[0] + eq_width > 300 or eq_position[1] + eq_height > 300:
                    collision = True

                for rect in bounding_rects:
                    if rect.collision(eq_bounding_rect):
                        collision = True
                        break

                if collision:
                    iterations += 1
                else:
                    bounding_rects.append(eq_bounding_rect)
                    eq_coords.append({"x1": eq_position[0], "y1": eq_position[1],
                                      "x2": eq_position[0] + eq_width, "y2": eq_position[1] + eq_height})

                    image_width, image_height = equation_image.size
                    equation_image = equation_image.resize(
                        (int(image_width * scale_factor), int(image_height * scale_factor)))

                    sheet_image.paste(
                        equation_image, (int(eq_position[0]), int(eq_position[1])), equation_image)
                    break

        # eq_coords must have max equations elements
        # if not, fill with empty coords
        while len(eq_coords) < self.max_equations_per_sheet:
            eq_coords.append({"x1": 0, "y1": 0, "x2": 0, "y2": 0})

        return (sheet_image, eq_coords)
