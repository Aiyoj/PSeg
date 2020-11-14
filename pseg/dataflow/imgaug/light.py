import cv2
import random
import numpy as np


class AugLight(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return data

        image = data["image"]

        value = random.randint(-30, 30)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image = np.array(hsv_image, dtype=np.float32)
        hsv_image[:, :, 2] += value
        hsv_image[hsv_image > 255] = 255
        hsv_image[hsv_image < 0] = 0
        hsv_image = np.array(hsv_image, dtype=np.uint8)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        data["image"] = image

        return data
