from .tokens import START_TOKEN, END_TOKEN
from .base_resnet_model import BaseResnetModel
from .equation_generator import EquationGenerator
from .feature_extractor import FeatureExtractor


def padded_equation_text(equation_text):
    return f'{START_TOKEN}{equation_text}{END_TOKEN}'


def ctc_equation_text(equation_text):
    return equation_text


class EquationPreprocessor:
    def __init__(self, equation_count, eq_dir):
        self.equation_count = equation_count
        self.feature_extractor = FeatureExtractor()
        self.equation_texts = {}
        self.equation_features = {}
        self.eq_dir = eq_dir

    def load_equations(self):
        print('Loading equation data...')
        equation_generator = EquationGenerator(self.eq_dir)
        equations = []
        if equation_generator.images_cached():
            print('Equation texts cached. Loading texts from cache...')
            equations = equation_generator.equations_from_cache()[
                :self.equation_count]
        else:
            print('Equation texts not cached. Generating images and texts...')
            for i in range(self.equation_count):
                equations.append(equation_generator.generate_equation_image())

        print(f'Loaded {len(equations)} equations.')

        for eq in equations:
            self.equation_texts[eq[0]] = ctc_equation_text(eq[1])

        # self.feature_extractor.load_features()
        # self.equation_features = self.feature_extractor.features
