from .tokens import MAX_EQUATION_TEXT_LENGTH
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from PIL import Image

import pandas as pd
import numpy as np

# http://man.hubwiz.com/docset/TensorFlow.docset/Contents/Resources/Documents/api_docs/python/tf/keras/backend/ctc_batch_cost.html


def fetch_and_preprocess_eq_image(eq_id):
    eq_image = Image.open(f'./equation_parser/data/images/{eq_id}.bmp')
    eq_image = np.array(eq_image.resize((100, 100)))
    return eq_image


class CtcDataGenerator:
    def __init__(self, vocab_size, equation_texts, tokenizer):
        self.vocab_size = vocab_size
        self.equation_texts = equation_texts
        self.batch_size = len(self.equation_texts.items())
        self.tokenizer = tokenizer

    def full_dataset(self):
        X1, y, input_length, label_length = list(), list(), list(), list()

        input_length = np.ones((self.batch_size, 1)) * 40
        label_length = np.zeros((self.batch_size, 1))

        for eq_id, equation_text in self.equation_texts.items():
            eq_image = fetch_and_preprocess_eq_image(eq_id)
            X1.append(eq_image)
            # img_to_predict = img_to_predict / 127.5
            # img_to_predict = img_to_predict - 1.0
            # encode the sequence
            sequence = self.tokenizer.texts_to_sequences([equation_text])[0]
            lbl_len = len(sequence)

            y.append(sequence)
            label_length.append(lbl_len)

        inputs = {
            'img_input': X1,
            'ground_truth_labels': y,
            'input_length': input_length,
            'label_length': label_length,
        }
        outputs = {'ctc': np.zeros([self.batch_size])}

        return (inputs, outputs)
