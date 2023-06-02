from tensorflow.keras import mixed_precision
from multiprocessing import freeze_support
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
import numpy as np
import sys
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os

from equation_analyzer.equation_finder.equation_finder import EquationFinder
from equation_analyzer.equation_finder.equation_sheet_generator import EquationSheetGenerator
from equation_analyzer.equation_finder.equation_sheet_decorator import EquationSheetDecorator
from equation_analyzer.equation_finder.equation_sheet_processor import EquationSheetProcessor

from equation_analyzer.equation_parser.equation_parser import EquationParser
# from equation_parser.equation_parser_simple import EquationParserSimple
from equation_analyzer.equation_parser.equation_generator import EquationGenerator
from equation_analyzer.equation_parser.caption_model import CaptionModel
from equation_analyzer.equation_parser.equation_tokenizer import EquationTokenizer
from equation_analyzer.equation_parser.equation_preprocessor import EquationPreprocessor

from equation_analyzer.equation_analyzer import EquationAnalyzer
from equation_analyzer.equation_parser.constants import MAX_EQUATION_TEXT_LENGTH
import tensorflow as tf

# Enable XLA
# tf.config.optimizer.set_jit(True)

# Enable mixed precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy('mixed_float16')

TRAIN = False
TEST = False
VIZ = False

if "train" in str(sys.argv[1]).lower():
    TRAIN = True
elif "test" in str(sys.argv[1]).lower():
    TEST = True
else:
    VIZ = True

TEST_IMAGES_DIR = './data/images'
CUSTOM_IMAGES_DIR = './data/images_custom'
CUSTOM_EQ_DIR = './data/equations_custom'


def run_eq_finder():
    esp = EquationSheetProcessor()
    # for i in range(5):
    for filename in os.listdir(CUSTOM_IMAGES_DIR):
        if not filename.endswith('jpg'):
            continue
        img_file = os.path.join(CUSTOM_IMAGES_DIR, filename)
        # test_image, eq_box = EquationSheetGenerator().dirty_sheet_with_equation(True)
        test_image = Image.open(img_file)
        test_image = test_image.resize((224, 224), Image.BICUBIC)

        # for i in range(1):
        #     eq_boxes.append(EquationSheetDecorator.add_equation(
        #         equation_sheet_image, eq_boxes))

        img, pred_box = esp.find_equation(test_image)

        fig, ax = plt.subplots()
        ax.imshow(test_image)

        if pred_box is None:
            continue

        width, height = pred_box.size()
        ax.add_patch(Rectangle(pred_box.topLeft, width, height,
                               fill=False, edgecolor="r"))

        # width, height = eq_box.size()
        # ax.add_patch(Rectangle(eq_box.topLeft, width, height,
        #                        fill=False, edgecolor="b"))

        # print('Inferred vs ground truth IOU: ', pred_box.iou(eq_box))
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


def generate_desc(model, tokenizer, photo):
    in_text = 's'
    photo = np.expand_dims(photo, axis=0)
    for i in range(MAX_EQUATION_TEXT_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences(
            [sequence], maxlen=MAX_EQUATION_TEXT_LENGTH, padding='post')
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
        vocab_size = len(tokenizer.word_index) + 2
        # # feature_extractor = FeatureExtractor()
        # # feature_extractor.load_features()
        # caption_model = CaptionModel(vocab_size)
        # caption_model.load_model()
        ep = EquationParser()
        model = CaptionModel(vocab_size, tokenizer, False)
        # model.create_model()
        model.load_model()
        for i in range(5):
            eq_id, tokens = EquationGenerator(
                './equation_analyzer/equation_parser/data/images_test').generate_equation_image()
            img_path = f'./equation_analyzer/equation_parser/data/images_test/{eq_id}.bmp'
            predicted_desc = ep.test_model(model.model, img_path, tokens)
            # eq_image_features = feature_extractor.features_from_image(eq_image)
            # print(eq_image_features)
            eq_image = Image.open(img_path)
            plt.figure()
            plt.imshow(eq_image)
            plt.text(0, -5, predicted_desc, fontsize=15)
        # for filename in os.listdir(CUSTOM_EQ_DIR):
        #     img_file = os.path.join(CUSTOM_EQ_DIR, filename)
        #     predicted_desc = ep.test_model(
        #         model.model, img_file, '452/158+256/124=789/371')
        #     eq_image = Image.open(img_file)
        #     plt.figure()
        #     plt.imshow(eq_image)
        #     plt.text(0, -5, predicted_desc, fontsize=15)
        plt.show()
    elif VIZ:
        for i in range(10):
            plt.figure(i)
            eq_id, tokens = EquationGenerator(
                './equation_analyzer/equation_parser/data/images_viz').generate_equation_image()
            img = Image.open(
                f'./equation_analyzer/equation_parser/data/images_viz/{eq_id}.bmp')
            # img = img.resize((150, 38))
            plt.text(0, -5, tokens, fontsize=15)
            plt.imshow(img)
        plt.show()


def viz_sheets():
    for i in range(5):
        equation_sheet_image, eq_box = EquationSheetGenerator().clean_sheet_with_equation()
        fig, ax = plt.subplots()
        width, height = eq_box.size()
        ax.add_patch(Rectangle(eq_box.topLeft, width, height,
                               fill=False, edgecolor="b"))
        ax.imshow(equation_sheet_image)
    plt.show()


def viz_custom_images():
    for filename in os.listdir(CUSTOM_IMAGES_DIR):
        if not filename.endswith('jpg'):
            continue
        img_file = os.path.join(CUSTOM_IMAGES_DIR, filename)

        img_idx = filename.split('.')[0]
        eq_box = EquationSheetGenerator().custom_image_eq_box(img_idx)
        eq_sheet_image = Image.open(img_file)
        fig, ax = plt.subplots()
        width, height = eq_box.size()
        ax.add_patch(Rectangle(eq_box.topLeft, width, height,
                               fill=False, edgecolor="b"))
        ax.set_title(filename)
        ax.imshow(eq_sheet_image)
    plt.show()


def main():
    # run_eq_parser()
    # viz_sheets()
    # EquationSheetProcessor()
    # run_eq_finder()
    EquationAnalyzer().start_stream()
    # viz_custom_images()


if __name__ == '__main__':
    freeze_support()
    data = main()
