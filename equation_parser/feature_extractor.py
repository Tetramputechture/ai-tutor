import os
import PIL
import pickle
import numpy as np

from tensorflow.keras import applications

FEATURES_PATH = './equation_parser/data/features.p'
IMAGES_PATH = './equation_parser/data/'


class FeatureExtractor:
    def __init__(self):
        self.model = applications.Xception(
            include_top=False, pooling='avg')
        self.features = {}

    def load_features(self):
        print('Loading equation image features...')
        if os.path.isfile(FEATURES_PATH):
            print('Equation image features cached. Loading features from cache...')
            self.features = pickle.load(open(FEATURES_PATH, 'rb'))
            return

        print('Equation image features not cached. Generating features...')
        self.generate_features()

    def generate_features(self):
        for filename in os.listdir(IMAGES_PATH):
            if not filename.endswith('.bmp'):
                continue

            # open image and resize to 299x299
            image = PIL.Image.open(os.path.join(IMAGES_PATH, filename))
            image = image.resize((299, 299))

            # xception preprocess per-pixel algorithm
            image = np.expand_dims(image, axis=0)
            image = image / 127.5
            image = image - 1.0

            feature = self.model.predict(image)[0]
            self.features[filename.split('.')[0]] = feature

        print('Features generated. Saving features...')
        self.save_features()

    def save_features(self):
        pickle.dump(self.features, open(FEATURES_PATH, 'wb'))
        print('Features saved to ', FEATURES_PATH)
