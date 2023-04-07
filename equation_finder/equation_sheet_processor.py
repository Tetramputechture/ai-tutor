from tensorflow.keras.preprocessing import image
import math

from PIL import Image, ImageDraw

from .equation_box import EquationBox
from .equation_finder import EquationFinder


def is_valid(eq_box):
    return abs(eq_box.bottomRight[0] - eq_box.topLeft[0]) >= 30 and \
        abs(eq_box.bottomRight[0] - eq_box.topLeft[0]) <= 310 and \
        abs(eq_box.bottomRight[1] - eq_box.topLeft[1]) >= 15 and \
        abs(eq_box.bottomRight[1] - eq_box.topLeft[1]) <= 200


class EquationSheetProcessor:
    def __init__(self):
        self.equation_finder = EquationFinder()
        self.equation_finder.load_model()

    def find_equations(self, sheet_image):
        equations = []
        new_sheet_image = sheet_image.copy()
        draw = ImageDraw.Draw(new_sheet_image)

        previous_inferred_box = None

        sheet_image_data = image.img_to_array(
            new_sheet_image.convert('RGB'))
        inferred_box = self.equation_finder.infer_from_model(
            sheet_image_data)

        print('Inferred: ', inferred_box)

        if is_valid(inferred_box) and inferred_box != previous_inferred_box:
            equations.append(inferred_box)
            previous_inferred_box = inferred_box
            cropped_center = [
                inferred_box.topLeft[0],
                inferred_box.topLeft[1],
                inferred_box.bottomRight[0],
                inferred_box.bottomRight[1]
            ]
            draw.rectangle(cropped_center, fill="white")

        return (new_sheet_image, equations)
