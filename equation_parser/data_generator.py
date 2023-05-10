from .tokens import MAX_EQUATION_TEXT_LENGTH
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from PIL import Image

import pandas as pd
import numpy as np


class DataGenerator:
    def __init__(self, vocab_size, equation_texts, tokenizer):
        self.vocab_size = vocab_size
        self.equation_texts = equation_texts
        # self.equation_features = equation_features
        self.tokenizer = tokenizer

    def full_dataset(self):
        X1, X2, y = list(), list(), list()

        for eq_id, equation_text in self.equation_texts.items():
            # equation_feature = self.equation_features[eq_id][0]
            # print(equation_text)

            eq_image = Image.open(f'./equation_parser/data/images/{eq_id}.bmp')
            eq_image = np.array(eq_image.resize((300, 300)))
            # img_to_predict = img_to_predict / 127.5
            # img_to_predict = img_to_predict - 1.0
            # encode the sequence
            sequence = self.tokenizer.texts_to_sequences([equation_text])[0]
            print(sequence)

            # print('Equation text: ', equation_text)

            # split one sequence into multiple X, y pairs
            for i in range(1, len(sequence)):
                # split into input and output pairs
                in_seq, out_seq = sequence[:i], sequence[i]

                # pad input sequence
                in_seq = pad_sequences(
                    [in_seq], maxlen=MAX_EQUATION_TEXT_LENGTH, padding='post', dtype='int32')[0]

                # print('X1 (equation feature):')
                # print(equation_feature)

                # print('X2 (input sequence):')
                # print(in_seq)
                # print(tokenizer.sequences_to_texts([in_seq]))

                # print('Y (token to predict):')
                # print(tokenizer.sequences_to_texts([[out_seq]]))

                # encode output sequence
                out_seq = to_categorical(
                    [out_seq], num_classes=self.vocab_size)[0]

                # store
                X1.append(eq_image)
                X2.append(in_seq)
                y.append(out_seq)

        return np.array(X1), np.array(X2), np.array(y)

    def save_data(self):
        pandas_data = {'feature': [], 'x2_str': [], 'y_str': []}

        features, input_texts, output_tokens = self.full_dataset()

        for idx, feature in enumerate(features):
            # print('Equation text:', equation_tex)
            x2_str = self.tokenizer.sequences_to_texts([input_texts[idx]])
            y_str = self.word_for_id(np.argmax(output_tokens[idx]))
            pandas_data['feature'].append(feature)
            pandas_data['x2_str'].append(x2_str)
            pandas_data['y_str'].append(y_str)

        pd.DataFrame(pandas_data).to_csv(
            './equation_parser/full_dataset.csv', index=False)

    def word_for_id(self, integer):
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None
