from multiprocessing import freeze_support

from equation_finder.equation_finder import EquationFinder


def main():
    ef = EquationFinder()
    ef.load_model()
    ef.show_validation()


if __name__ == '__main__':
    freeze_support()
    main()
