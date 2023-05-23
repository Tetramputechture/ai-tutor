from .equation_generator import EquationGenerator
import multiprocessing as mp


class EquationPreprocessor:
    def __init__(self, equation_count, eq_dir):
        self.equation_count = equation_count
        self.equation_generator = EquationGenerator(eq_dir)
        self.equation_texts = {}

    def load_equations(self):
        print('Loading equation data...')

        equations = []
        if self.equation_generator.images_cached():
            print('Equation texts cached. Loading texts from cache...')
            equations = self.equation_generator.equations_from_cache()[
                :self.equation_count]
        else:
            print('Equation texts not cached. Generating images and texts...')
            pool = mp.Pool()
            manager = mp.Manager()

            equations = manager.list()
            [pool.apply_async(self.append_equation, args=[equations])
             for i in range(self.equation_count)]
            pool.close()
            pool.join()

        equations = list(equations)
        print(f'Loaded {len(equations)} equations.')

        for eq in equations:
            self.equation_texts[eq[0]] = eq[1]

    def append_equation(self, equations):
        equations.append(self.equation_generator.generate_equation_image())
