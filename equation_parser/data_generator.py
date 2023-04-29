from .tokens import MAX_EQUATION_TEXT_LENGTH, VOCAB_SIZE
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np


class DataGenerator:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def data_generator(self, equation_texts, equation_features, tokenizer):
        while True:
            for eq_id, equation_text in equation_texts.items():
                equation_feature = equation_features[eq_id]
                img_features, input_text, output_token = self.create_sequences(
                    tokenizer, equation_text, equation_feature)
                yield [[img_features, input_text], output_token]

    def create_sequences(self, tokenizer, equation_text, equation_feature):
        X1, X2, y = list(), list(), list()

        # encode the sequence
        sequence = tokenizer.texts_to_sequences([equation_text])[0]

        # split one sequence into multiple X, y pairs
        for i in range(1, len(sequence)):
            # split into input and output pairs
            in_seq, out_seq = sequence[:i], sequence[i]

            # pad input sequence
            in_seq = pad_sequences(
                [in_seq], maxlen=MAX_EQUATION_TEXT_LENGTH)[0]

            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]

            # store
            X1.append(equation_feature)
            X2.append(in_seq)
            y.append(out_seq)

        return np.array(X1), np.array(X2), np.array(y)
