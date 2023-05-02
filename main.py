from multiprocessing import freeze_support
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from equation_finder.equation_finder import EquationFinder
from equation_finder.equation_sheet_generator import EquationSheetGenerator
from equation_finder.equation_sheet_decorator import EquationSheetDecorator
from equation_finder.equation_sheet_processor import EquationSheetProcessor

from equation_parser.equation_parser import EquationParser
from equation_parser.equation_parser_simple import EquationParserSimple
from equation_parser.equation_generator import EquationGenerator


def run_eq_finder():
    esp = EquationSheetProcessor()
    for _ in range(5):
        equation_sheet_image, eq_box = EquationSheetGenerator().clean_sheet_with_equation()
        # for i in range(1):
        #     eq_boxes.append(EquationSheetDecorator.add_equation(
        #         equation_sheet_image, eq_boxes))

        (img, predictions) = esp.find_equations(equation_sheet_image)

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.imshow(equation_sheet_image)

        for box in predictions:
            width, height = box.size()
            ax.add_patch(Rectangle(box.topLeft, width, height,
                                   fill=False, edgecolor="r"))

        width, height = eq_box.size()
        ax.add_patch(Rectangle(eq_box.topLeft, width, height,
                               fill=False, edgecolor="b"))

        if len(predictions) > 0:
            print('Inferred vs ground truth IOU: ', predictions[0].iou(eq_box))
        # for i in range(0, 1900, 50):
        #     sheet = EquationSheetGenerator().sheet_from_file(
        #         f'./data/equation-sheet-images/eq-sheet-{i}.bmp')
        #     fig, ax = plt.subplots()
        #     ax.imshow(sheet[0])
        #     box = sheet[1]
        #     width, height = box.size()
        #     ax.add_patch(Rectangle(box.topLeft, width, height,
        #                            fill=False, edgecolor="r"))
    plt.show()
    # eq = EquationFinder()
    # eq.load_model()
    # eq.show_validation()


def run_eq_parser():
    # EquationParser().train_model()
    for i in range(5):
        plt.figure(i)
        eq_id, tokens = EquationGenerator().generate_equation_image()
        plt.imshow(Image.open(f'./equation_parser/data/{eq_id}.bmp'))
    plt.show()


def main():
    run_eq_parser()


if __name__ == '__main__':
    freeze_support()
    main()
