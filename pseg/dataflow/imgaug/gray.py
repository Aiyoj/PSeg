import cv2
import random
import numpy as np


class AugGray(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        assert "image" in data and "label" in data, "`image` and `label` in data is required by this process"
        if random.random() < self.p:
            return data
        rgb_image = data["image"]
        label = data["label"]

        gray_image = np.zeros_like(rgb_image)
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        gray_image[:, :, 0] = gray
        gray_image[:, :, 1] = gray
        gray_image[:, :, 2] = gray

        mask = np.zeros_like(label)
        mask[label > 50] = 1

        new_image = gray_image * np.expand_dims(mask, -1) + rgb_image * np.expand_dims((1 - mask), -1)
        data["image"] = new_image

        return data
