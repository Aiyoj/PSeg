import cv2
import random
import numpy as np


class AugBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        assert "image" in data, "`image` in data is required by this process"

        if random.random() < 1 - self.p:
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


class AugMotionBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        assert "image" in data and "label" in data, "`image` and `label` in data is required by this process"

        if random.random() < 1 - self.p:
            return data

        image = data["image"]
        label = data["label"]

        degree = random.randint(5, 30)
        angle = random.randint(0, 360)

        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree

        img_blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        label_blurred = cv2.filter2D(label, -1, motion_blur_kernel)

        cv2.normalize(img_blurred, img_blurred, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(label_blurred, label_blurred, 0, 255, cv2.NORM_MINMAX)

        data["image"] = img_blurred
        data["label"] = label_blurred

        return data
