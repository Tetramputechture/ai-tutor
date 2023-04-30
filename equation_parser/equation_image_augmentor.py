import tensorflow as tf
from PIL import Image


class EquationImageAugmentor:
    def augment_equation_image(equation_image):
        seed = (1, 0)
        new_image = tf.image.stateless_random_brightness(
            equation_image, max_delta=0.5, seed=seed)
        new_image = tf.image.stateless_random_contrast(
            new_image, lower=0.3, upper=0.6, seed=seed)
