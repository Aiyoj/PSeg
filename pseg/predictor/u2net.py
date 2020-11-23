import cv2
import torch
import numpy as np

from addict import Dict

from pseg.algorithm.u2net import U2Net


class U2NetPredictor(object):
    def __init__(self, config=None):
        self.cfg = Dict(config)

        self.model_name = self.cfg.get("model_name", "small")
        self.min_side_len = self.cfg.get("min_side_len", 320)
        self.resume = self.cfg.get("resume", False)

        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = U2Net({"model_name": self.model_name})
        if self.resume:
            self.model.load_state_dict(torch.load(self.resume, map_location=self.device))
            print(f"load model from {self.resume} success !")

        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def normalize_v2(im):
        img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im -= img_mean
        im /= img_std

        im = im.transpose((2, 0, 1))
        return im

    def resize(self, im):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        min_side_len = self.min_side_len
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the min side
        if max(resize_h, resize_w) > min_side_len:
            # if resize_h > resize_w:
            #     ratio = float(min_side_len) / resize_w
            # else:
            #     ratio = float(min_side_len) / resize_h

            if resize_h > resize_w:
                ratio = float(min_side_len) / resize_h
            else:
                ratio = float(min_side_len) / resize_w

        # # limit the min side
        # if min(resize_h, resize_w) > min_side_len:
        #     if resize_h > resize_w:
        #         ratio = float(min_side_len) / resize_w
        #     else:
        #         ratio = float(min_side_len) / resize_h

        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)
        if resize_h % 32 == 0:
            resize_h = resize_h
        elif resize_h // 32 <= 1:
            resize_h = 32
        else:
            resize_h = (resize_h // 32 - 1) * 32
        if resize_w % 32 == 0:
            resize_w = resize_w
        elif resize_w // 32 <= 1:
            resize_w = 32
        else:
            resize_w = (resize_w // 32 - 1) * 32
        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            im = cv2.resize(im, (int(resize_w), int(resize_h)))
        except:
            print(im.shape, resize_w, resize_h)
            return None, (None, None)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def resize_v2(self, im):
        h, w = im.shape[:2]
        if h > w:
            resize_h, resize_w = self.min_side_len * h / w, self.min_side_len
        else:
            resize_h, resize_w = self.min_side_len, self.min_side_len * w / h

        resize_h, resize_w = int(resize_h), int(resize_w)

        imm = cv2.resize(im, (resize_w, resize_h))

        return imm

    def predict(self, image):
        im = self.resize_v2(image)
        im = self.normalize_v2(im)
        im = np.expand_dims(im, 0)
        im = torch.from_numpy(im)
        im = im.to(self.device)

        with torch.no_grad():
            ret = self.model(im)

            d1 = ret["d0"]

            def normPRED(d):
                ma = torch.max(d)
                mi = torch.min(d)

                dn = (d - mi) / (ma - mi)

                return dn

            pred = d1[:, 0, :, :]
            pred = normPRED(pred)

            pred = pred.detach().cpu().numpy()
            pred = np.transpose(pred, (1, 2, 0)).squeeze()

            return pred
