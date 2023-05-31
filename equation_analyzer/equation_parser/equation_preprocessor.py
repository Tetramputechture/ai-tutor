from .equation_generator import EquationGenerator
import multiprocessing as mp
import uuid
import os
import csv
import tqdm

from ctypes import c_int

TOKENS_FILENAME = 'tokens'
TOKENS_HEADERS = ['eq_id', 'tokens']


class EquationPreprocessor:
    def __init__(self, equation_count, eq_dir):
        self.equation_count = equation_count
        self.equation_generator = EquationGenerator(eq_dir)
        self.eq_dir = eq_dir
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
            # eq_counter = mp.Value(c_int, 0)
            pool = mp.Pool()
            manager = mp.Manager()
            self.equations = manager.list()

            os.makedirs(self.eq_dir)

            # [pool.apply_async(self.append_equation, args=[equations])
            #  for i in range(self.equation_count)]
            # equations = pool.imap_unordered(self.equation_generator.generate_equation_image,
            #                                 range(self.equation_count))
            # pool.close()
            # pool.join()

            for _ in tqdm.tqdm(pool.imap_unordered(self.append_equation, range(self.equation_count)), total=self.equation_count):
                pass

            self.equations = list(self.equations)

            with open(f'{self.eq_dir}/{TOKENS_FILENAME}.csv', 'a', newline='', encoding='utf-8') as tokens_file:
                writer = csv.writer(tokens_file)
                writer.writerow(TOKENS_HEADERS)

            with open(f'{self.eq_dir}/{TOKENS_FILENAME}.csv', 'a', newline='', encoding='utf-8') as tokens_file:
                for equation in self.equations:
                    eq_id, eq_tokens = equation
                    writer = csv.writer(tokens_file)
                    writer.writerow([eq_id, eq_tokens])

        print(f'Loaded {len(self.equations)} equations.')

        for eq in self.equations:
            self.equation_texts[eq[0]] = eq[1]

    def append_equation(self, _):
        self.equations.append(
            self.equation_generator.generate_equation_image())
        # with self.val.get_lock():
        #     self.val.value += 1
        # print('aaa')
        # with counter.get_lock():
        #     counter.value += 1
        #     percent_complete = (counter.value / self.equation_count) * 100
        #     print(f'{percent_complete}%', flush=True)
