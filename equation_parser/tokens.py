import string
import numpy as np

START_TOKEN = 'START'
END_TOKEN = 'END'
PAD_TOKEN = 'PAD'

TOKENS = list(string.digits + '+=/') + [PAD_TOKEN, START_TOKEN, END_TOKEN]
VOCAB_SIZE = len(TOKENS)

TOKENS_ONEHOT = []
for idx, t in enumerate(TOKENS):
    zeroes = np.zeros(len(TOKENS))
    zeroes[idx] = 1
    TOKENS_ONEHOT.append(zeroes)

MAX_EQUATION_TEXT_LENGTH = 13  # 6 numbers + 3 fractions + plus, equals + START + END

CONTEXT_WINDOW_LENGTH = 5


def pad_tokens(tokens, max_length, encoded_pad_token):
    result = []
    tokens_len = len(tokens)
    for i in range(max_length):
        if i < tokens_len:
            result.append(tokens[i])
        else:
            result.append(encoded_pad_token)

    return np.array(result)
