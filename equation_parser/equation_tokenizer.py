import os
import pickle
from keras.preprocessing.text import Tokenizer

TOKENIZER_PATH = './equation_parser/data/equation_tokenizer.p'


class EquationTokenizer:
    def __init__(self, equation_texts):
        self.equation_texts = equation_texts

    def equation_texts_list(self):
        return list(self.equation_texts.values())

    def load_tokenizer(self):
        if os.path.isfile(TOKENIZER_PATH):
            return pickle.load(open(TOKENIZER_PATH, 'rb'))

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.equation_texts_list())
        pickle.dump(tokenizer, open(TOKENIZER_PATH, 'wb'))
        return tokenizer
