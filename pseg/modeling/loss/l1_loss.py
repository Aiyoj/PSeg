import torch
import torch.nn as nn


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, gt):
        b, c, h, w = gt.shape
        loss = (torch.abs(pred - gt)).sum() / (h * w)
        return loss
