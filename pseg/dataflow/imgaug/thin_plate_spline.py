import cv2
import random
import numpy as np


class ThinPlateSpline(object):
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
        tps = cv2.createThinPlateSplineShapeTransformer()
        sshape = np.array(
            [
                [0 + bias[0], 0 + bias[1]],
                [height + bias[2], 0 + bias[3]],
                [0 + bias[4], width + bias[5]],
                [height + bias[6], width + bias[7]]
            ],
            np.float32
        )
        tshape = np.array(
            [
                [0 + bias[8], 0 + bias[9]],
                [height + bias[10], 0 + bias[11]],
                [0 + bias[12], width + bias[13]],
                [height + bias[14], width + bias[15]]
            ],
            np.float32
        )
        sshape = sshape.reshape((1, -1, 2))
        tshape = tshape.reshape((1, -1, 2))
        matches = list()
        matches.append(cv2.DMatch(0, 0, 0))
        matches.append(cv2.DMatch(1, 1, 0))
        matches.append(cv2.DMatch(2, 2, 0))
        matches.append(cv2.DMatch(3, 3, 0))

        tps.estimateTransformation(tshape, sshape, matches)
        image_thin_plate_spline = tps.warpImage(image)
        label_thin_plate_spline = tps.warpImage(label)

        data["image"] = image_thin_plate_spline
        data["label"] = label_thin_plate_spline

        return data
