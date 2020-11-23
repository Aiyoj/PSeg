import cv2
import random


class FlipLR(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        assert "image" in data and "label" in data, "`image` and `label` in data is required by this process"

        if random.random() < self.p:
            return data

        image = data["image"]
        label = data["label"]

        h_image = cv2.flip(image, 1)
        h_label = cv2.flip(label, 1)

        data["image"] = h_image
        data["label"] = h_label

        return data
