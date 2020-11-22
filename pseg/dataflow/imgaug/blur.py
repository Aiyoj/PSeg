import cv2
import random


class AugBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        assert "image" in data, "`image` in data is required by this process"

        if random.random() < self.p:
            return data

        image = data["image"]

        select = random.random()
        if select < 0.3:
            ks = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ks, ks), 0)
        elif select < 0.6:
            ks = random.choice([3, 5])
            image = cv2.medianBlur(image, ks)
        else:
            ks = random.choice([3, 5])
            image = cv2.blur(image, (ks, ks))

        data["image"] = image

        return data
