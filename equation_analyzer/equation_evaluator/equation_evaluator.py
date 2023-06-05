import sympy
import re

FRAC_REGEX = '\+|='


def eq_string_to_numbers(eq_string):
    fractions = [frac.split('/') for frac in re.split(FRAC_REGEX, eq_string)]
    if len(fractions) == 0:
        return []
    return list(filter(lambda x: x >= 0,
                       [int(num or -1) for nums in fractions for num in nums]))


class EquationEvaluator:
    def __init__(self, eq_string):
        self.eq_numbers = eq_string_to_numbers(eq_string)

    def is_correct(self):
        if len(self.eq_numbers) != 6:
            return False

        a = sympy.Rational(self.eq_numbers[0], self.eq_numbers[1])
        b = sympy.Rational(self.eq_numbers[2], self.eq_numbers[3])
        c = sympy.Rational(self.eq_numbers[4], self.eq_numbers[5])

        correct_solution = a + b

        # print(f'Original equation: {a}+{b}={c}')
        # print(f'Correct equation: {a}+{b}={correct_solution}')
        return correct_solution == c
