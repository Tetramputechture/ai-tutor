from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt

from PIL import Image, ImageDraw

from .equation_box import EquationBox
from .equation_finder import EquationFinder

PREDICTED_EQ_IMAGE_PADDING = 5


def is_valid(eq_box):
    return abs(eq_box.bottomRight[0] - eq_box.topLeft[0]) >= 30 and \
        abs(eq_box.bottomRight[0] - eq_box.topLeft[0]) <= 224 and \
        abs(eq_box.bottomRight[1] - eq_box.topLeft[1]) >= 5 and \
        abs(eq_box.bottomRight[1] - eq_box.topLeft[1]) <= 224


class EquationSheetProcessor:
    def __init__(self):
        self.equation_finder = EquationFinder()
        self.equation_finder.load_model()

    def find_equation(self, sheet_image):
        new_sheet_image = sheet_image.copy()

        new_sheet_image = new_sheet_image.convert('L').convert('RGB')
        sheet_image_data = image.img_to_array(new_sheet_image)
        inferred_box = self.equation_finder.infer_from_model(
            sheet_image_data)

        # print(inferred_box)

        if is_valid(inferred_box):
            # pad box by a few pixels
            padded_box = EquationBox(
                (inferred_box.topLeft[0] - PREDICTED_EQ_IMAGE_PADDING,
                 inferred_box.topLeft[1] - PREDICTED_EQ_IMAGE_PADDING),
                (inferred_box.bottomRight[0] + PREDICTED_EQ_IMAGE_PADDING,
                 inferred_box.bottomRight[1] + PREDICTED_EQ_IMAGE_PADDING)
            )
            eq_image = new_sheet_image.crop((
                padded_box.topLeft[0],
                padded_box.topLeft[1],
                padded_box.bottomRight[0],
                padded_box.bottomRight[1]
            ))
            return (eq_image, padded_box)

        return (None, None)
