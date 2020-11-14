import cv2
import numpy as np


class RandomCrop(object):
    def __init__(
            self,
            size=(640, 640)
    ):
        self.size = size

    def __call__(self, data):
        image = data["image"]
        label = data["label"]

        h, w = image.shape[:2]

        idx = np.where(label > 0)
        if len(idx[0]) > 0:
            min_y = idx[0].min()
            max_y = idx[0].max()
        else:
            min_y = 0
            max_y = h - 1

        if len(idx[1]) > 0:
            min_x = idx[1].min()
            max_x = idx[1].max()
        else:
            min_x = 0
            max_x = w - 1

        y1 = np.random.randint(0, min_y + 1)
        y2 = np.random.randint(max_y, h)

        x1 = np.random.randint(0, min_x + 1)
        x2 = np.random.randint(max_x, w)

        crop_image = image[y1: y2, x1: x2]
        crop_label = label[y1: y2, x1: x2]
        # data["crop_label"] = label[min_y:max_y, min_x:max_x]

        crop_h, crop_w = crop_image.shape[:2]

        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        new_h = int(crop_h * scale)
        new_w = int(crop_w * scale)

        pad_image = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        pad_image[:new_h, :new_w] = cv2.resize(crop_image, (new_w, new_h))

        pad_label = np.zeros((self.size[1], self.size[0]), np.uint8)
        pad_label[:new_h, :new_w] = cv2.resize(crop_label, (new_w, new_h))

        data["image"] = pad_image
        data["label"] = pad_label

        return data
