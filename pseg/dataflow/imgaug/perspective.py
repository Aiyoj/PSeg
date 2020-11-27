import cv2
import random
import numpy as np


class Perspective(object):
    def __init__(self, p=0.5, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data):
        assert "image" in data and "label" in data, "`image` and `label` in data is required by this process"

        if random.random() < 1 - self.p:
            return data

        image = data["image"]
        label = data["label"]

        height, width = image.shape[:2]

        bias = np.random.randint(-int(height * self.ratio), int(width * self.ratio), 16)
        pts1 = np.float32(
            [
                [0 + bias[0], 0 + bias[1]],
                [height + bias[2], 0 + bias[3]],
                [0 + bias[4], width + bias[5]],
                [height + bias[6], width + bias[7]]
            ]
        )
        pts2 = np.float32(
            [
                [0 + bias[8], 0 + bias[9]],
                [height + bias[10], 0 + bias[11]],
                [0 + bias[12], width + bias[13]],
                [height + bias[14], width + bias[15]]
            ]
        )
        M = cv2.getPerspectiveTransform(pts1, pts2)
        image_perspective = cv2.warpPerspective(image, M, (width, height))
        label_perspective = cv2.warpPerspective(label, M, (width, height))

        data["image"] = image_perspective
        data["label"] = label_perspective

        return data
