import string
import numpy as np

START_TOKEN = 's'
END_TOKEN = 'e'
PAD_TOKEN = 'p'

TOKENS = list(string.digits + '+=/') + [PAD_TOKEN, START_TOKEN, END_TOKEN]

VOCAB_SIZE = len(TOKENS)
TOKENS_ONEHOT = []
for idx, t in enumerate(TOKENS):
    zeroes = np.zeros(len(TOKENS))
    zeroes[idx] = 1
    TOKENS_ONEHOT.append(zeroes)

# 18 numbers + 3 fractions + plus, equals
MAX_EQUATION_TEXT_LENGTH = 23

RNN_TIMESTEPS = 37


# MAX_EQUATIN_TEXT_LENGTH = 13

CONTEXT_WINDOW_LENGTH = 5
