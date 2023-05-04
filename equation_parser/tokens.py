import string
import numpy as np

START_TOKEN = 's'
END_TOKEN = 'e'
PAD_TOKEN = 'p'

TOKENS = list(string.digits + '+=/') + [PAD_TOKEN, START_TOKEN, END_TOKEN]

# 18 numbers + 3 fractions + plus, equals + S + E
MAX_EQUATION_TEXT_LENGTH = 25

# MAX_EQUATIN_TEXT_LENGTH = 13

CONTEXT_WINDOW_LENGTH = 5
