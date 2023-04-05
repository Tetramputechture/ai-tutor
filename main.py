from multiprocessing import freeze_support
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from equation_finder.equation_finder import EquationFinder
from equation_finder.equation_sheet_generator import EquationSheetGenerator
from equation_finder.equation_sheet_decorator import EquationSheetDecorator
from equation_finder.equation_sheet_processor import EquationSheetProcessor


def main():
    eq_boxes = []
    equation_sheet_image, eq_box = EquationSheetGenerator().clean_sheet_with_equation()
    eq_boxes.append(eq_box)
    # for i in range(1):
    #     eq_boxes.append(EquationSheetDecorator.add_equation(
    #         equation_sheet_image, eq_boxes))

    esp = EquationSheetProcessor(equation_sheet_image)
    # (img, predictions) = esp.find_equations()
    # print(predictions)

    # fig, ax = plt.subplots(2, 1)
    # ax[0].imshow(img)
    # ax[1].imshow(equation_sheet_image)

    # for box in predictions:
    #     width, height = box.size()
    #     ax[1].add_patch(Rectangle(box.topLeft, width, height,
    #                               fill=False, edgecolor="r"))

    # width, height = eq_box.size()
    # ax[1].add_patch(Rectangle(eq_box.topLeft, width, height,
    #                           fill=False, edgecolor="b"))

    # if len(predictions) > 0:
    #     print('Inferred vs ground truth IOU: ', eq_box.iou(predictions[0]))
    # plt.show()
    # eq = EquationFinder()
    # eq.load_model()
    # eq.show_validation()


if __name__ == '__main__':
    freeze_support()
    main()
