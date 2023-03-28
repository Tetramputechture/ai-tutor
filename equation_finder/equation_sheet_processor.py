from tensorflow.keras.preprocessing import image
import math

from PIL import Image, ImageDraw

from .equation_box import EquationBox
from .equation_finder import EquationFinder


def is_zero(eq_box):
    return abs(eq_box.bottomRight[0] - eq_box.topLeft[0]) <= 60 and \
        abs(eq_box.bottomRight[1] - eq_box.topLeft[1]) <= 30


class EquationSheetProcessor:
    def __init__(self, sheet_image):
        self.sheet_image = sheet_image
        self.equation_finder = EquationFinder()
        self.equation_finder.load_model()

    def find_equations(self):
        equations = []
        new_sheet_image = self.sheet_image.copy()
        draw = ImageDraw.Draw(new_sheet_image)

        while True:
            sheet_image_data = image.img_to_array(
                new_sheet_image.convert('RGB'))
            inferred_box = self.equation_finder.infer_from_model(
                sheet_image_data)

            print("Inferred:", inferred_box.topLeft, inferred_box.bottomRight)

            if is_zero(inferred_box):
                break
            else:
                equations.append(inferred_box)
                cropped_center = [
                    inferred_box.topLeft[0] + 10,
                    inferred_box.topLeft[1] + 10,
                    inferred_box.bottomRight[0] - 10,
                    inferred_box.bottomRight[1] - 10
                ]
                draw.rectangle(cropped_center, fill="white")

        return equations
