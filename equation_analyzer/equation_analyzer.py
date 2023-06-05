from .equation_finder.equation_sheet_processor import EquationSheetProcessor
from .equation_parser.equation_parser import EquationParser
from .equation_parser.caption_model import CaptionModel
from .equation_parser.equation_tokenizer import EquationTokenizer
from .equation_evaluator.equation_evaluator import EquationEvaluator
import numpy as np
from datetime import datetime
import cv2
from PIL import Image


EQUATION_FINDER_SIZE = (224, 224)

PREDICTION_SECONDS = 3


class EquationAnalyzer:
    def __init__(self):
        self.eq_localizer = EquationSheetProcessor()
        self.eq_parser = EquationParser()
        self.tokenizer = EquationTokenizer().load_tokenizer()
        vocab_size = len(self.tokenizer.word_index) + 2

        self.caption_model = CaptionModel(vocab_size, self.tokenizer, False)
        # self.caption_model.create_model()
        self.caption_model.load_model()

    def start_stream(self):
        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            raise IOError('Cannot open webcam')

        # import time
        # time to seconds (int)
        # every  5 seconds,
        # add each prediction per frame to array
        # display most common prediction
        # or
    # if same prediction repeated 3 times

        last_prediction_time = datetime.now()
        most_common_prediction = ''
        predictions = {}
        first_prediction = True

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

                # cv2.imshow('eq image', eq_image)

                # tokenize the equation image
                tokens = self.eq_parser.test_model_raw_img(self.tokenizer,
                                                           self.caption_model.model, eq_image)

                if len(tokens) > 0:
                    # print('Predicted tokens: ', tokens)
                    if not tokens in predictions:
                        predictions[tokens] = 1
                    else:
                        predictions[tokens] += 1

                    # if it has been more than N seconds since last prediction time,
                    # calculate most common prediction, display, and reset predictions
                    # and set last prediction time to now
                    now = datetime.now()

                    if (now - last_prediction_time).total_seconds() >= PREDICTION_SECONDS:
                        most_common_prediction = max(
                            predictions, key=predictions.get)
                        last_prediction_time = datetime.now()
                        predictions = {}

                if len(most_common_prediction) > 0:
                    text_pos = scaled_eq_box.shift((0, -10))
                    analyzer = EquationEvaluator(most_common_prediction)
                    if analyzer.is_correct():
                        text_color = (0, 255, 0)
                    else:
                        text_color = (0, 0, 255)
                    if not first_prediction:
                        cv2.putText(frame, most_common_prediction, text_pos.topLeft,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, 3)
                    else:
                        first_prediction = False

                # cv2.putText(frame, tokens, (10, 40),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 2)

                # cv2.rectangle(
                #     resized_frame, eq_box.topLeft, eq_box.bottomRight, color=(255, 0, 0), thickness=2)
                cv2.rectangle(frame, scaled_eq_box.topLeft, scaled_eq_box.bottomRight, color=(
                    255, 0, 0), thickness=2)
                # cv2.imshow('Localized EQ', resized_frame)

            cv2.imshow('Webcam', frame)

            c = cv2.waitKey(1)
            if c == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
