import os
import cv2
import torch
import random
import numpy as np

from pseg.dataflow.imgaug.affine import Affine
from pseg.dataflow.imgaug.blur import AugMotionBlur, AugBlur
from pseg.dataflow.imgaug.noise import AugNoise
from pseg.dataflow.imgaug.flip import FlipLR
from pseg.dataflow.imgaug.gray import AugGray
from pseg.dataflow.imgaug.hsv import AugHSV
from pseg.dataflow.imgaug.iaa import IaaAugment
from pseg.dataflow.imgaug.normalize import Normalize
from pseg.dataflow.label_generator.make_dist_map import MakeDistMap
from pseg.dataflow.imgaug.crop import RandomCrop
from pseg.dataflow.raw import DataFromList
from pseg.dataflow.parallel import MapAndBatchData
from pseg.dataflow.common import BatchData


class DataLoader(object):
    def __init__(self, data_dir: list, data_list: list, batch_size: int, num_worker: int = 1, buffer_size: int = 32,
                 shuffle: bool = True, seed: int = None, remainder: bool = True, is_training: bool = True,
                 pre_processes: list = None):
        self.data_dir = data_dir
        self.data_list = data_list
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.seed = seed
        self.remainder = remainder
        self.is_training = is_training
        self.pre_processes = pre_processes

        self.image_paths = []
        self.gt_paths = []

        if self.seed is not None:
            random.seed(self.seed)

        for i in range(len(self.data_dir)):
            with open(self.data_list[i], "r") as fid:
                row_list = fid.readlines()
                image_path = [
                    os.path.join(self.data_dir[i], timg.strip().split("\t")[0]) for timg in row_list
                ]
                gt_path = [
                    os.path.join(self.data_dir[i], timg.strip().split("\t")[1]) for timg in row_list
                ]

                self.image_paths += image_path
                self.gt_paths += gt_path

        self.data_list = [(img_path, gt_path) for img_path, gt_path in zip(self.image_paths, self.gt_paths)]

        if self.pre_processes is None:
            self.pre_processes = [
                {"type": "AugHSV", "args": {"p": 0.5}},
                {"type": "AugNoise", "args": {"p": 0.5}},
                {"type": "AugGray", "args": {"p": 0.5}},
                {"type": "AugBlur", "args": {"p": 0.5}},
                # {"type": "AugMotionBlur", "args": {"p": 0.5}},
                {"type": "FlipLR", "args": {"p": 0.5}},
                {"type": "Affine", "args": {"rotate": [-20, 20]}},
                {"type": "RandomCrop", "args": {}},
                {"type": "Normalize", "args": {}},
                {"type": "MakeDistMap", "args": {}}
            ]
        self.augs = []
        for aug in self.pre_processes:
            if "args" not in aug:
                args = {}
            else:
                args = aug["args"]
            if isinstance(args, dict):
                cls = eval(aug["type"])(**args)
            else:
                cls = eval(aug["type"])(args)
            self.augs.append(cls)
        ds = DataFromList(self.data_list, shuffle=self.shuffle, seed=self.seed)
        ds = BatchData(ds, self.batch_size, use_list=True, remainder=self.remainder)
        self.ds = MapAndBatchData(ds, self.num_worker, self.map_fn, 1, buffer_size=self.buffer_size)

        self.ds.start()

    def map_fn(self, datapoints):
        targets = {
            "image": [],
            "ori_image": [],
            "ori_label": [],
            "shape": [],
            "filename": [],
            "label": [],
            "dist_map": []
        }
        size = random.choice([320])
        for dp in zip(datapoints[0], datapoints[1]):
            image_path, gt_path = dp
            try:
                im = cv2.imread(image_path, cv2.IMREAD_COLOR)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            except Exception:
                print(image_path)
                continue

            dp_targets = {}
            dp_targets["label"] = gt
            dp_targets["image"] = im
            # dp_targets["ori_image"] = im
            # dp_targets["ori_label"] = gt
            dp_targets["shape"] = [im.shape[0], im.shape[1]]
            dp_targets["filename"] = image_path

            for aug in self.augs:
                if isinstance(aug, RandomCrop):
                    aug.size = [size, size]
                dp_targets = aug(dp_targets)

            targets["image"].append(dp_targets["image"])
            targets["label"].append(dp_targets["label"])
            targets["shape"].append(dp_targets["shape"])
            targets["filename"].append(dp_targets["filename"])

            if "dist_map" in dp_targets.keys():
                targets["dist_map"].append(dp_targets["dist_map"])
            else:
                targets.pop("dist_map")

            if "ori_image" in dp_targets.keys():
                targets["ori_image"].append(dp_targets["ori_image"])
            else:
                targets.pop("ori_image")

            if "ori_label" in dp_targets.keys():
                targets["ori_label"].append(dp_targets["ori_label"])
            else:
                targets.pop("ori_label")

        return targets

    def __iter__(self):
        for j, targets in enumerate(self.ds):
            for key, value in targets.items():
                try:
                    if key in ["image", "label", "dist_map"]:
                        targets[key] = torch.from_numpy(np.array(value[0]))
                    else:
                        targets[key] = value[0]
                except Exception:
                    for a in value[0]:
                        print(key, a.shape)
            yield targets

    def __len__(self):
        return len(self.ds)
