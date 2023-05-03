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

            image = PIL.Image.open(os.path.join(IMAGES_PATH, filename))
            feature = self.features_from_image(image)
            self.features[filename.split('.')[0]] = feature

        print('Features generated. Saving features...')
        self.save_features()

    def features_from_image(self, image):
        img_to_predict = image.resize((299, 299))

        # xception preprocess per-pixel algorithm
        img_to_predict = np.expand_dims(img_to_predict, axis=0)
        img_to_predict = img_to_predict / 127.5
        img_to_predict = img_to_predict - 1.0

        return self.model.predict(img_to_predict)

    def save_features(self):
        pickle.dump(self.features, open(FEATURES_PATH, 'wb'))
        print('Features saved to ', FEATURES_PATH)
