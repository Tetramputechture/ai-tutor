import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets, models, applications, optimizers, backend
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Input, Reshape, Bidirectional, Lambda, LSTM
from tensorflow.keras.utils import plot_model
from keras import backend as K


from .constants import RNN_TIMESTEPS, MAX_EQUATION_TEXT_LENGTH, EQ_IMAGE_HEIGHT, EQ_IMAGE_WIDTH
from .base_resnet_model import BaseResnetModel
from .base_conv_model import BaseConvModel

MODEL_PATH = './equation_analyzer/equation_parser/caption_model.h5'
MODEL_IMG_PATH = './equation_analyzer/equation_parser/equation_parser.png'


def ctc_loss_function(args):
    """
    CTC loss function takes the values passed from the model returns the CTC loss using Keras Backend ctc_batch_cost function
    """
    y_pred, y_true, input_length, label_length = args
    # since the first couple outputs of the RNN tend to be garbage we need to discard them, found this from other CRNN approaches
    # I Tried by including these outputs but the results turned out to be very bad and got very low accuracies on prediction
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


class CaptionModel:
    def __init__(self, vocab_size, tokenizer, train=True):
        self.vocab_size = vocab_size
        self.base_resnet_model = BaseResnetModel()
        self.base_conv_model = BaseConvModel()
        self.tokenizer = tokenizer
        self.train = train

    def create_model(self):
        # self.base_resnet_model.load_model()
        self.base_conv_model.load_model()

        LSTM_UNITS = 256
        N = int(MAX_EQUATION_TEXT_LENGTH * RNN_TIMESTEPS)

        model_input = Input(
            shape=(EQ_IMAGE_WIDTH, EQ_IMAGE_HEIGHT, 1), name='img_input')
        model = self.base_conv_model.model(model_input)
        # model = Dense(N, activation='relu')(model)
        model = Reshape(target_shape=(
            (RNN_TIMESTEPS, 1024)))(model)
        model = Dense(64, activation='relu')(model)
        model = Bidirectional(LSTM(
            LSTM_UNITS, return_sequences=True), merge_mode='sum')(model)
        model = Bidirectional(LSTM(
            LSTM_UNITS, return_sequences=True), merge_mode='concat')(model)
        model = Dense(self.vocab_size)(model)
        model_output = Activation(
            'softmax', name='softmax')(model)

        labels = Input(name='ground_truth_labels', shape=[
                       MAX_EQUATION_TEXT_LENGTH], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(ctc_loss_function, output_shape=(1,), name='ctc')(
            [model_output, labels, input_length, label_length])

        if self.train:
            self.model = models.Model(
                inputs=[model_input, labels, input_length, label_length], outputs=loss_out)

            self.model.compile(optimizer='adam',
                               loss={'ctc': lambda y_true, y_pred: y_pred})

            plot_model(self.model, to_file=MODEL_IMG_PATH, show_shapes=True)

            self.test_func = K.function([model_input], [model_output])

            return (model_input, model_output, self.model)
        else:
            self.model = models.Model(
                inputs=[model_input], outputs=model_output)

    def load_model(self):
        if self.model_cached():
            print('Model cached. Loading model...')
            self.create_model()
            self.model.load_weights(
                './equation_analyzer/equation_parser/caption_model.h5')
            # self.model = models.load_model(MODEL_PATH, compile=False)
            # self.model.compile(optimizer='adam',
            #                    loss='categorical_crossentropy',
            #                    metrics=['accuracy'])
        else:
            print('Model not cached. Creating model...')
            self.create_model()

    def model_cached(self):
        return os.path.exists(MODEL_PATH)

    def save_model(self):
        self.model.save(MODEL_PATH)
