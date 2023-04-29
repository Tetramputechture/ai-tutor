from .tokens import START_TOKEN, END_TOKEN
from .base_resnet_model import BaseResnetModel
from .equation_generator import EquationGenerator


def padded_equation_text(equation_text):
    return f'{START_TOKEN} {equation_text} {END_TOKEN}'


class EquationPreprocessor:
    def __init__(self, equation_count):
        self.equation_count = equation_count
        self.base_resnet_model = BaseResnetModel()
        self.equation_texts = {}
        self.equation_features = {}

    def load_equations(self):
        self.base_resnet_model.load_model()
        generator = EquationGenerator(self.base_resnet_model.model)
        equations = []
        if generator.images_cached():
            equations = generator.equations_from_cache()[
                :self.equation_count]
        else:
            for i in range(self.equation_count):
                equations.append(generator.generate_equation_image())

        for eq in equations:
            self.equation_texts[eq[3]] = padded_equation_text(eq[1])
            self.equation_features[eq[3]] = eq[2]

        return (self.equation_texts, self.equation_features)
