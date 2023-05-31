from .equation_finder.equation_sheet_processor import EquationSheetProcessor
from .equation_parser.equation_parser import EquationParser
from .equation_parser.caption_model import CaptionModel
from .equation_parser.equation_tokenizer import EquationTokenizer
import numpy as np

import cv2
from PIL import Image


EQUATION_FINDER_SIZE = (224, 224)


class EquationAnalyzer:
    def __init__(self):
        self.eq_localizer = EquationSheetProcessor()
        self.eq_parser = EquationParser()
        tokenizer = EquationTokenizer().load_tokenizer()
        vocab_size = len(tokenizer.word_index) + 2

        self.caption_model = CaptionModel(vocab_size, tokenizer, False)
        self.caption_model.create_model()
        self.caption_model.load_model()

    def start_stream(self):
        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            raise IOError('Cannot open webcam')

        tokenizer = EquationTokenizer().load_tokenizer()

        while True:
            ret, frame = cap.read()

            original_frame_res = frame.shape

            scale = (original_frame_res[1] / EQUATION_FINDER_SIZE[0],
                     original_frame_res[0] / EQUATION_FINDER_SIZE[1])

            # resize frame to EQ finder size
            resized_frame = cv2.resize(
                frame, EQUATION_FINDER_SIZE, interpolation=cv2.INTER_AREA)

            # # cv2.imshow(resized_frame)

            # convert cv2 to pil
            color_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(color_frame)

            # localize the equation image
            found_img, eq_box = self.eq_localizer.find_equation(pil_frame)

            if eq_box is not None:
                # print('Found equation: ', eq_box)

                # rescale the localized equation box and then get the image data from that
                scaled_eq_box = eq_box.scale(scale)

                # print('Original EQ box: ', eq_box)
                # print('Scaled box: ', scaled_eq_box)

                eq_image = frame[scaled_eq_box.topLeft[1]:scaled_eq_box.bottomRight[1],
                                 scaled_eq_box.topLeft[0]:scaled_eq_box.bottomRight[0]]

                color_eq_image = cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB)
                pil_eq_image = Image.fromarray(color_eq_image)

                cv2.rectangle(frame, scaled_eq_box.topLeft, scaled_eq_box.bottomRight, color=(
                    0, 0, 255), thickness=2)

                # eq_image = pil_frame_img.crop(
                #     (scaled_eq_box.topLeft[0],
                #      scaled_eq_box.topLeft[1],
                #      scaled_eq_box.bottomRight[0],
                #      scaled_eq_box.bottomRight[1])
                # )

                # tokenize the equation image
                tokens = self.eq_parser.test_model_raw_img(tokenizer,
                                                           self.caption_model.model, eq_image)

                if len(tokens) > 0:
                    # print('Predicted tokens: ', tokens)

                    cv2.putText(frame, tokens, (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 2)

                cv2.rectangle(
                    resized_frame, eq_box.topLeft, eq_box.bottomRight, color=(255, 0, 0), thickness=2)
                cv2.imshow('Localized EQ', resized_frame)

            cv2.imshow('Webcam', frame)

            c = cv2.waitKey(1)
            if c == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
