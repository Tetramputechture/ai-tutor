from multiprocessing import freeze_support
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from equation_finder.equation_finder import EquationFinder
from equation_finder.equation_sheet_generator import EquationSheetGenerator
from equation_finder.equation_sheet_decorator import EquationSheetDecorator
from equation_finder.equation_sheet_processor import EquationSheetProcessor


def main():
    # eq_boxes = []
    # equation_sheet_image, eq_box = EquationSheetGenerator().clean_sheet_with_equation()
    # eq_boxes.append(eq_box)
    # for i in range(2):
    #     eq_boxes.append(EquationSheetDecorator.add_equation(
    #         equation_sheet_image, eq_boxes))

    # esp = EquationSheetProcessor(equation_sheet_image)
    # predictions = esp.find_equations()

    # fig, ax = plt.subplots()
    # ax.imshow(equation_sheet_image)

    # for box in predictions:
    #     width, height = box.size()
    #     ax.add_patch(Rectangle(box.topLeft, width, height,
    #                            fill=False, edgecolor="r"))

    # plt.show()
    eq = EquationFinder()
    eq.load_model()
    eq.show_validation()


if __name__ == '__main__':
    freeze_support()
    main()
