import os
import pickle
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import TextVectorization
from tensorflow.data.Data import from_tensor_slices

from .tokens import MAX_EQUATION_TEXT_LENGTH


TOKENIZER_PATH = './equation_parser/equation_tokenizer.p'


class EquationTokenizer:
    def __init__(self, equation_texts=None):
        self.equation_texts = equation_texts

    def equation_texts_list(self):
        return list(self.equation_texts.values())

    def vectorizer(self):
        vectorize_layer = TextVectorization(
            max_tokens=None,
            standardize='lower_and_strip_punctuation',
            split='character',
            ngrams=None,
            output_mode='int',
            output_sequence_length=MAX_EQUATION_TEXT_LENGTH,
            pad_to_max_tokens=False,
            vocabulary=None,
            idf_weights=None,
            sparse=False,
            ragged=False,
            encoding='utf-8',
            **kwargs
        )
        text_dataset = from_tensor_slices(self.equation_texts_list())
        vectorize_layer.adapt(text_dataset.batch(64))
        return vectorize_layer

    def int_vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return self.vectorizer()(text), label
