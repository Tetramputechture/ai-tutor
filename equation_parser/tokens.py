import string
import numpy as np

TOKENS = list(string.digits + '+=/') + ['PAD', 'START', 'END']
VOCAB_SIZE = len(TOKENS)

TOKENS_ONEHOT = []
for idx, t in enumerate(TOKENS):
    zeroes = np.zeros(len(TOKENS))
    zeroes[idx] = 1
    TOKENS_ONEHOT.append(zeroes)

MAX_EQ_TOKEN_LENGTH = 25  # 18 digits + 3 fractions + plus, equals + START + END
MIN_EQ_TOKEN_LENGTH = 13  # 1/1 + 1/1 = 1/1 + START + END

CONTEXT_WINDOW_LENGTH = 5
