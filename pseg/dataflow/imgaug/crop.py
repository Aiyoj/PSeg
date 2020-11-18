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

        main_image = image[y1: y2, x1: x2]
        main_label = label[y1: y2, x1: x2]

        main_h, main_w = main_image.shape[:2]

        if main_h > main_w:
            resize_h, resize_w = self.size[0] * main_h / main_w, self.size[1]
        else:
            resize_h, resize_w = self.size[0], self.size[1] * main_w / main_h

        resize_h, resize_w = int(resize_h), int(resize_w)

        flag = np.random.randint(0, 4)
        if flag == 0:
            resized_image = cv2.resize(main_image, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)
        elif flag == 1:
            resized_image = cv2.resize(main_image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        elif flag == 2:
            resized_image = cv2.resize(main_image, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
        else:
            resized_image = cv2.resize(main_image, (resize_w, resize_h), interpolation=cv2.INTER_LANCZOS4)

        resized_label = cv2.resize(main_label, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)

        top = np.random.randint(0, resize_h - self.size[0] + 1)
        left = np.random.randint(0, resize_w - self.size[1] + 1)

        new_image = resized_image[top: top + self.size[0], left: left + self.size[1]]
        new_label = resized_label[top: top + self.size[0], left: left + self.size[1]]

        data["image"] = new_image
        data["label"] = new_label

        return data
