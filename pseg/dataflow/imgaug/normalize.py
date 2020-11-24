import numpy as np


class Normalize(object):
    def __init__(self):
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, data):
        assert "image" in data and "label" in data, "`image` and `label` in data is required by this process"
        image = data["image"]
        image = image.astype(np.float32, copy=False)
        image /= 255.
        image -= self.img_mean
        image /= self.img_std
        image = np.transpose(image, [2, 0, 1])
        data["image"] = image

        label = data["label"]
        label = label.astype(np.float32, copy=False)
        label /= 255.
        label = np.expand_dims(label, 0)
        data["label"] = label

        return data
