import torch
import torch.nn as nn


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        self.idc = [0]

    def forward(self, probs, dist_maps):
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = torch.einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss
