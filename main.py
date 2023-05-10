from multiprocessing import freeze_support
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import sys
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

from equation_finder.equation_finder import EquationFinder
from equation_finder.equation_sheet_generator import EquationSheetGenerator
from equation_finder.equation_sheet_decorator import EquationSheetDecorator
from equation_finder.equation_sheet_processor import EquationSheetProcessor

from equation_parser.equation_parser import EquationParser
# from equation_parser.equation_parser_simple import EquationParserSimple
from equation_parser.equation_generator import EquationGenerator
from equation_parser.caption_model import CaptionModel
from equation_parser.feature_extractor import FeatureExtractor
from equation_parser.equation_tokenizer import EquationTokenizer
from equation_parser.equation_preprocessor import EquationPreprocessor

from equation_parser.tokens import MAX_EQUATION_TEXT_LENGTH

from equation_parser.data_generator import DataGenerator

TRAIN = False
TEST = False
VIZ = False

if "train" in str(sys.argv[1]).lower():
    TRAIN = True
elif "test" in str(sys.argv[1]).lower():
    TEST = True
else:
    VIZ = True


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


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, photo):
    in_text = 's'
    for i in range(MAX_EQUATION_TEXT_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences(
            [sequence], maxlen=MAX_EQUATION_TEXT_LENGTH, padding='post', value='p')
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += word
        if word == 'e':
            break
    return in_text


def run_eq_parser():
    if TRAIN:
        EquationParser().train_model()
    elif TEST:
        tokenizer = EquationTokenizer().load_tokenizer()
        vocab_size = len(tokenizer.word_index) + 1
        feature_extractor = FeatureExtractor()
        feature_extractor.load_features()
        caption_model = CaptionModel(vocab_size)
        caption_model.load_model()
        for i in range(5):
            eq_id, tokens = EquationGenerator().generate_equation_image()
            eq_image = Image.open(f'./equation_parser/data/images/{eq_id}.bmp')
            eq_image_features = feature_extractor.features_from_image(eq_image)
            # print(eq_image_features)
            predicted_desc = generate_desc(
                caption_model.model, eq_image_features)

            plt.figure()
            plt.imshow(eq_image)
            plt.text(0, 0, predicted_desc)
        plt.show()
    elif VIZ:
        for i in range(2):
            plt.figure(i)
            eq_id, tokens = EquationGenerator().generate_equation_image()
            plt.imshow(Image.open(
                f'./equation_parser/data/images/{eq_id}.bmp'))
        plt.show()


def visualize_data():
    equation_preprocessor = EquationPreprocessor(
        10)
    equation_preprocessor.load_equations()
    equation_texts = equation_preprocessor.equation_texts
    equation_features = equation_preprocessor.equation_features

    tokenizer = EquationTokenizer(equation_texts).load_tokenizer()
    vocab_size = len(tokenizer.word_index) + 1
    generator = DataGenerator(vocab_size)

    full_dataset = generator.data_viz_generator(
        equation_texts, equation_features, tokenizer)

    pandas_data = {'eq_id': [], 'x2_str': [], 'y_str': []}

    for datum in full_dataset:
        pandas_data['eq_id'].append(datum['eq_id'])
        pandas_data['x2_str'].append(datum['x2_str'])
        pandas_data['y_str'].append(datum['y_str'])

    pd.DataFrame(pandas_data).to_csv(
        './equation_parser/full_dataset.csv', index=False)


def main():
    run_eq_parser()


if __name__ == '__main__':
    freeze_support()
    main()
