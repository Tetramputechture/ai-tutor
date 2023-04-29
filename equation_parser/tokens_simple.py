import string
import numpy as np

TOKENS = list(string.digits + '+=/') + ['PAD']

TOKENS_ONEHOT = []
for idx, t in enumerate(TOKENS):
    zeroes = np.zeros(len(TOKENS))
    zeroes[idx] = 1
    TOKENS_ONEHOT.append(zeroes)

MAX_EQUATION_TEXT_LENGTH = 23  # 18 digits + 3 fractions + plus, equals
