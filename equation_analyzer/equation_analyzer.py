from .equation_finder.equation_sheet_processor import EquationSheetProcessor
from .equation_parser.equation_parser import EquationParser
from .equation_parser.caption_model import CaptionlModel
from .equation_parser.equation_tokenizer import EquationTokenizer

import cv2

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

    def start_stream():
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise IOError('Cannot open webcam')

        while True:
            ret, frame = cap.read()

            original_frame_res = frame.shape

            scale = (EQUATION_FINDER_SIZE[0] / original_frame_res[0],
                     EQUATION_FINDER_SIZE[1] / original_frame_res[1])

            # resize frame to EQ finder size
            resized_frame = cv2.resize(resized_frame, EQUATION_FINDER_SIZE)

            # localize the equation image
            _, eq_box = self.eq_localizer.find_equation(resized_frame)

            if eq_image is None:
                continue

            # rescale the localized equation box and then get the image data from that
            scaled_eq_box = eq_box.scale(scale)

            eq_image = frame[scaled_eq_box.topLeft[1]:scaled_eq_box.bottomRight[1],
                             scaled_eq_box.topLeft[0]:scaled_eq_box.bottomRight[0]]

            # tokenize the equation image
            tokens = self.eq_parser.test_modeL_raw_img(
                self.caption_model.model, eq_image, tokens)

            print('Found equation!')
            print('Predicted tokens: ', tokens)

            cv2.rectangle(frame, (scaled_eq_box.topLeft[0], scaled_eq_box.topLeft[1]), (
                scaled_eq_box.bottomRight[0], scaled_eq_box.bottomRight[1]), color=(255, 0, 0), thickness=2)

            cv2.putText(frame, tokens, (0, 0),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 2)

            cv2.imshow('Webcam', frame)

        cap.release()
        cv2.destroyAllWindows()
