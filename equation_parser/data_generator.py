from .tokens import MAX_EQUATION_TEXT_LENGTH
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np


class DataGenerator:
    def __init__(self, vocab_size, equation_texts, equation_features, tokenizer):
        self.vocab_size = vocab_size
        self.equation_texts = equation_texts
        self.equation_features = equation_features
        self.tokenizer = tokenizer

    def full_dataset(self):
        X1, X2, y = list(), list(), list()

        for eq_id, equation_text in self.equation_texts.items():
            equation_feature = self.equation_features[eq_id][0]
            # print(equation_text)

            # encode the sequence
            sequence = self.tokenizer.texts_to_sequences([equation_text])[0]

            # print('Equation text: ', equation_text)

            # split one sequence into multiple X, y pairs
            for i in range(1, len(sequence)):
                # split into input and output pairs
                in_seq, out_seq = sequence[:i], sequence[i]

                # pad input sequence
                in_seq = pad_sequences(
                    [in_seq], maxlen=MAX_EQUATION_TEXT_LENGTH, padding='post', value=-1.0, dtype='float32')[0]

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
                X1.append(equation_feature)
                X2.append(in_seq)
                y.append(out_seq)

        return np.array(X1), np.array(X2), np.array(y)

    def data_generator(self, equation_texts, equation_features, tokenizer):
        while True:
            for eq_id, equation_text in equation_texts.items():
                equation_feature = equation_features[eq_id][0]
                img_features, input_texts, output_tokens = self.create_sequences(
                    tokenizer, equation_text, equation_feature)
                yield [[img_features, input_texts], output_tokens]

    def full_data(self, equation_texts, equation_features, tokenizer):
        x, y = list(), list()

        for eq_id, equation_text in equation_texts.items():
            equation_feature = equation_features[eq_id][0]
            img_features, input_texts, output_tokens = self.create_sequences(
                tokenizer, equation_text, equation_feature)

            for idx, text in enumerate(input_texts):
                x.append([img_features[idx], text])
                y.append(output_tokens[idx])

        return np.array(x), np.array(y)

        # yield [[img_features, input_text], output_token]

    def data_viz_generator(self, equation_texts, equation_features, tokenizer):
        # while True:
        # print(equation_texts.items())
        full_dataset = []
        for eq_id, equation_text in equation_texts.items():
            # print('Equation text:', equation_text)
            equation_feature = equation_features[eq_id][0]

            # print('Equation features:', equation_feature)
            img_features, input_texts, output_tokens = self.create_sequences(
                tokenizer, equation_text, equation_feature)

            for idx, texts in enumerate(input_texts):
                x2_str = tokenizer.sequences_to_texts([input_texts[idx]])
                y_str = tokenizer.sequences_to_texts([[output_tokens[idx]]])
                full_dataset.append(
                    {'eq_id': eq_id, 'x2_str': x2_str, 'y_str': y_str})

        return full_dataset

    def create_sequences(self, tokenizer, equation_text, equation_feature):
        X1, X2, y = list(), list(), list()

        # print(equation_text)

        # encode the sequence
        sequence = tokenizer.texts_to_sequences([equation_text])[0]

        # print('Equation text: ', equation_text)

        # split one sequence into multiple X, y pairs
        for i in range(1, len(sequence)):
            # split into input and output pairs
            in_seq, out_seq = sequence[:i], sequence[i]

            # pad input sequence
            in_seq = pad_sequences(
                [in_seq], maxlen=MAX_EQUATION_TEXT_LENGTH, padding='post', value=-1.0, dtype='float32')[0]

            # print('X1 (equation feature):')
            # print(equation_feature)

            # print('X2 (input sequence):')
            # print(in_seq)
            # print(tokenizer.sequences_to_texts([in_seq]))

            # print('Y (token to predict):')
            # print(tokenizer.sequences_to_texts([[out_seq]]))

            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]

            # store
            X1.append(equation_feature)
            X2.append(in_seq)
            y.append(out_seq)

        return np.array(X1), np.array(X2), np.array(y)
