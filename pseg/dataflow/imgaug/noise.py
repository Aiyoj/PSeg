import random
import numpy as np


class AugNoise(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        assert "image" in data, "`image` in data is required by this process"

        if random.random() < self.p:
            return data

        image = data["image"]

        mu = 0
        sigma = random.random() * 10.0
        image = np.array(image, dtype=np.float32)
        image += np.random.normal(mu, sigma, image.shape)
        image[image > 255] = 255
        image[image < 0] = 0

        data["image"] = image

        return data
