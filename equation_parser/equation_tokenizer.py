import os
import pickle
from keras.preprocessing.text import Tokenizer

TOKENIZER_PATH = './equation_parser/equation_tokenizer.p'


class EquationTokenizer:
    def __init__(self, equation_texts=None):
        self.equation_texts = equation_texts

    def equation_texts_list(self):
        return list(self.equation_texts.values())

    def load_tokenizer(self):
        print('Loading equation text tokenizer...')
        if os.path.isfile(TOKENIZER_PATH):
            print('Tokenizer cached. Loading tokenizer from cache...')
            return pickle.load(open(TOKENIZER_PATH, 'rb'))

        print('Tokenizer not cached. Fitting new tokenizer...')
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(self.equation_texts_list())

        print('Tokenizer fitted. Saving tokenizer to cache...')
        pickle.dump(tokenizer, open(TOKENIZER_PATH, 'wb'))
        print('Tokenizer saved.')

        return tokenizer
