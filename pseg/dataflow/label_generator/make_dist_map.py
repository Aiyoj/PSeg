import numpy as np

from scipy.ndimage import distance_transform_edt as distance


class MakeDistMap(object):
    def __init__(self):
        pass

    def gen_dist_map(self, label):
        """
        :param label: numpy array, [1, w, h]
        :return:
        """
        res = np.zeros_like(label, np.float32)

        posmask = np.zeros_like(label, np.float32)
        negmask = np.zeros_like(label, np.float32)

        posmask[label >= 0.2] = 1
        posmask[label < 0.2] = 0
        negmask[label >= 0.2] = 0
        negmask[label < 0.2] = 1

        if posmask.sum() >= 1:
            res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

        return res

    def __call__(self, data):
        label = data["label"]
        dist_map = self.gen_dist_map(label)

        data["dist_map"] = dist_map

        return data
