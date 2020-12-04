import cv2
import math
import torch
import numpy as np

from addict import Dict

from pseg.algorithm.u2net import U2Net


class U2NetPredictor(object):
    def __init__(self, config=None):
        self.cfg = Dict(config)

        self.model_config = self.cfg.get(
            "model_config", {
                "backbone": {"type": "U2NetBackbone", "args": {"in_ch": 3, "model_name": "large"}},
                "head": {"type": "U2NetHead", "args": {"out_ch": 1, "model_name": "large"}}
            }
        )
        self.min_side_len = self.cfg.get("min_side_len", 320)
        self.resume = self.cfg.get("resume", False)
        self.scale = self.cfg.get("scale", 32)
        self.key = self.cfg.get("key", "d0")
        self.strict = self.cfg.get("strict", True)

        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = U2Net(self.model_config)
        if self.resume:
            self.model.load_state_dict(torch.load(self.resume, map_location=self.device), strict=self.strict)
            print(f"load model from {self.resume} success !")

        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def normalize(im):
        img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im -= img_mean
        im /= img_std

        im = im.transpose((2, 0, 1))
        return im

    def resize(self, im):
        h, w = im.shape[:2]
        if h > w:
            resize_h, resize_w = self.min_side_len * h / w, self.min_side_len
        else:
            resize_h, resize_w = self.min_side_len, self.min_side_len * w / h

        padding_h, padding_w = resize_h, resize_w

        if h > w:
            if padding_h % self.scale == 0:
                padding_h = padding_h
            else:
                padding_h = math.ceil(padding_h / self.scale) * self.scale
        else:
            if padding_w % self.scale == 0:
                padding_w = padding_w
            else:
                padding_w = math.ceil(padding_w / self.scale) * self.scale

        resize_h, resize_w = int(resize_h), int(resize_w)
        padding_h, padding_w = int(padding_h), int(padding_w)

        padding_image = np.zeros((padding_h, padding_w, 3), np.uint8)
        # print(padding_image.shape)

        im = cv2.resize(im, (resize_w, resize_h))
        # print(im.shape)

        padding_image[:resize_h, :resize_w, :] = im

        return padding_image, (padding_h - resize_h, padding_w - resize_w)

    def predict(self, image):
        im, (pad_h, pad_w) = self.resize(image)
        im = self.normalize(im)
        im = np.expand_dims(im, 0)
        im = torch.from_numpy(im)
        im = im.to(self.device)

        with torch.no_grad():
            ret = self.model(im)

            d = ret[self.key]

            def normPRED(d):
                ma = torch.max(d)
                mi = torch.min(d)

                dn = (d - mi) / (ma - mi)

                return dn

            pred = d[:, 0, :, :]
            pred = normPRED(pred)

            pred = pred.detach().cpu().numpy()
            pred = np.transpose(pred, (1, 2, 0)).squeeze()

            return pred
