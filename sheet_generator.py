import random
import math
import numpy as np

import PIL

from bounding_rect import BoundingRect


class SheetGenerator:
    def __init__(self, original_sheet_image, original_eq_coords):
        self.original_sheet_image = original_sheet_image
        self.original_eq_coords = original_eq_coords

    def generate_sheet(self):
        new_image = PIL.Image.new(mode="RGBA", size=(
            self.original_sheet_image.size[0], self.original_sheet_image.size[1]), color=(255, 255, 255, 0))
        new_eq_coords = []
        for eq_coord in self.original_eq_coords:
            # make new bounding rect
            bounding_rect = BoundingRect.from_coords(
                (eq_coord['x1'], eq_coord['y1']),
                (eq_coord['x2'], eq_coord['y2']))
            new_eq_image = self.original_sheet_image.crop(
                (eq_coord['x1'], eq_coord['y1'],
                 eq_coord['x2'], eq_coord['y2']))
            new_eq_image = new_eq_image.convert('RGBA')
            # scale
            if random.uniform(0, 1) < 0.5:
                scale_factor = random.uniform(-0.1, 0.1) + 1
                bounding_rect = bounding_rect.scale(scale_factor)

                image_width, image_height = new_eq_image.size
                new_eq_image = new_eq_image.resize(
                    (int(image_width * scale_factor), int(image_height * scale_factor)))
            # shift
            if random.uniform(0, 1) < 0.5:
                shift_x = random.randint(-15, 15)
                shift_y = random.randint(-3, 3)
                bounding_rect = bounding_rect.shift((shift_x, shift_y))
            # rotate (todo)
            if random.uniform(1, 1) < 0.3:
                rotation_deg = random.randint(-5, 5)
                rotation_rad = rotation_deg * 0.017453

                new_eq_image = new_eq_image.rotate(
                    rotation_deg, PIL.Image.NEAREST, expand=1, fillcolor='white')
            fff = PIL.Image.new('RGBA', new_eq_image.size, (255,)*4)
            new_eq_image = PIL.Image.composite(new_eq_image, fff, new_eq_image)

            new_image.paste(
                new_eq_image, (int(bounding_rect.x), int(bounding_rect.y)))
            image_width, image_height = new_image.size
            new_eq_coords.append({"x1": bounding_rect.x, "y1": bounding_rect.y, "x2": bounding_rect.x +
                                 bounding_rect.width, "y2": bounding_rect.y + bounding_rect.height})
        return (new_image, new_eq_coords)
