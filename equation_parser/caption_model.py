import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, layers, models, applications, optimizers, backend
from tensorflow.keras.utils import plot_model

from .tokens import MAX_EQUATION_TEXT_LENGTH
from .base_resnet_model import BaseResnetModel
from .base_conv_model import BaseConvModel

MODEL_PATH = './equation_parser/caption_model.h5'
MODEL_IMG_PATH = './equation_parser/equation_parser.png'


def ctc_loss_function(args):
    """
    CTC loss function takes the values passed from the model returns the CTC loss using Keras Backend ctc_batch_cost function
    """
    y_pred, y_true, input_length, label_length = args
    # since the first couple outputs of the RNN tend to be garbage we need to discard them, found this from other CRNN approaches
    # I Tried by including these outputs but the results turned out to be very bad and got very low accuracies on prediction
    y_pred = y_pred[:, 2:, :]
    return backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


class CaptionModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.base_resnet_model = BaseResnetModel()
        self.base_conv_model = BaseConvModel()

    def decode_label(self, out):
        """
        Takes the predicted ouput matrix from the Model and returns the output text for the image
        """
        # out : (1, 42, 37)
        # discarding first 2 outputs of RNN as they tend to be garbage
        out_best = list(np.argmax(out[0, 2:], axis=1))

        out_best = [k for k, g in itertools.groupby(
            out_best)]  # remove overlap value

        outstr = words_from_labels(out_best)
        return outstr

    def create_model(self):
        self.base_resnet_model.load_model()

        UNITS_PER_TIMESTEP = 32
        LSTM_UNITS = 256
        N = int(MAX_EQUATION_TEXT_LENGTH * UNITS_PER_TIMESTEP)

        model_input = layers.Input(shape=(100, 100, 3), name='img_input')
        model = self.base_resnet_model.model(model_input)
        model = layers.Dense(N, activation='relu')(model)
        model = layers.Reshape(target_shape=(
            (MAX_EQUATION_TEXT_LENGTH, UNITS_PER_TIMESTEP)))(model)
        model = layers.Bidirectional(layers.LSTM(
            LSTM_UNITS, return_sequences=True), merge_mode='sum')(model)
        model = layers.Bidirectional(layers.LSTM(
            LSTM_UNITS, return_sequences=True), merge_mode='concat')(model)
        output = layers.Dense(self.vocab_size, activation='softmax')(model)

        labels = Input(name='ground_truth_labels', shape=[
                       MAX_EQUATION_TEXT_LENGTH], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(ctc_loss_function, output_shape=(1,), name='ctc')(
            [output, labels, input_length, label_length])

        self.model = models.Model(
            inputs=[model_input, labels, input_length, label_length], outputs=loss_out)

        self.model.compile(optimizer='adam',
                           loss='ctc',
                           metrics=['accuracy'])

        plot_model(self.model, to_file=MODEL_IMG_PATH, show_shapes=True)

    def load_model(self):
        if self.model_cached():
            print('Model cached. Loading model...')
            self.model = models.load_model(MODEL_PATH, compile=False)
            self.model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
        else:
            print('Model not cached. Creating model...')
            self.create_model()

    def model_cached(self):
        return os.path.exists(MODEL_PATH)

    def save_model(self):
        self.model.save(MODEL_PATH)
