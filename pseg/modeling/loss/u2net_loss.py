import torch.nn as nn

from collections import OrderedDict

from pseg.modeling.loss.dice_loss import DiceLoss
from pseg.modeling.loss.boundary_loss import BoundaryLoss
from pseg.modeling.loss.balance_cross_entropy_loss import BalanceCrossEntropyLoss


class U2NetLossV1(nn.Module):
    def __init__(self, eps=1e-6):
        super(U2NetLossV1, self).__init__()
        self.eps = eps
        self.dice_loss = DiceLoss(eps=self.eps)
        self.boundary_loss = BoundaryLoss()

    def forward(self, pred, batch, alpha=1):
        d0 = pred["d0"]
        d1 = pred["d1"]
        d2 = pred["d2"]
        d3 = pred["d3"]
        d4 = pred["d4"]
        d5 = pred["d5"]
        d6 = pred["d6"]

        label = batch["label"]
        dist_map = batch["dist_map"]

        d0_dice = self.dice_loss(d0, label)
        d1_dice = self.dice_loss(d1, label)
        d2_dice = self.dice_loss(d2, label)
        d3_dice = self.dice_loss(d3, label)
        d4_dice = self.dice_loss(d4, label)
        d5_dice = self.dice_loss(d5, label)
        d6_dice = self.dice_loss(d6, label)

        dice_loss = d0_dice + d1_dice + d2_dice + d3_dice + d4_dice + d5_dice + d6_dice

        d0_boundary = self.boundary_loss(d0, dist_map)
        d1_boundary = self.boundary_loss(d1, dist_map)
        d2_boundary = self.boundary_loss(d2, dist_map)
        d3_boundary = self.boundary_loss(d3, dist_map)
        d4_boundary = self.boundary_loss(d4, dist_map)
        d5_boundary = self.boundary_loss(d5, dist_map)
        d6_boundary = self.boundary_loss(d6, dist_map)

        boundary_loss = d0_boundary + d1_boundary + d2_boundary + d3_boundary + d4_boundary + d5_boundary + d6_boundary

        metrics = OrderedDict(
            d0_dice=d0_dice, d1_dice=d1_dice, d2_dice=d2_dice, d3_dice=d3_dice,
            d4_dice=d4_dice, d5_dice=d5_dice, d6_dice=d6_dice, dice_loss=dice_loss,
            d0_boundary=d0_boundary, d1_boundary=d1_boundary, d2_boundary=d2_boundary, d3_boundary=d3_boundary,
            d4_boundary=d4_boundary, d5_boundary=d5_boundary, d6_boundary=d6_boundary, boundary_loss=boundary_loss,
        )

        loss = alpha * dice_loss + (1 - alpha) * boundary_loss

        return loss, metrics


class U2NetLossV2(nn.Module):
    def __init__(self):
        super(U2NetLossV2, self).__init__()
        self.bce_loss = BalanceCrossEntropyLoss()

    def forward(self, pred, batch, alpha=1):
        d0 = pred["d0"]
        d1 = pred["d1"]
        d2 = pred["d2"]
        d3 = pred["d3"]
        d4 = pred["d4"]
        d5 = pred["d5"]
        d6 = pred["d6"]

        label = batch["label"]
        # dist_map = batch["dist_map"]

        d0_bce = self.bce_loss(d0, label)
        d1_bce = self.bce_loss(d1, label)
        d2_bce = self.bce_loss(d2, label)
        d3_bce = self.bce_loss(d3, label)
        d4_bce = self.bce_loss(d4, label)
        d5_bce = self.bce_loss(d5, label)
        d6_bce = self.bce_loss(d6, label)

        bce_loss = d0_bce + d1_bce + d2_bce + d3_bce + d4_bce + d5_bce + d6_bce

        metrics = OrderedDict(
            d0_bce=d0_bce, d1_bce=d1_bce, d2_bce=d2_bce, d3_bce=d3_bce,
            d4_bce=d4_bce, d5_bce=d5_bce, d6_bce=d6_bce, bce_loss=bce_loss
        )

        loss = bce_loss

        return loss, metrics


class U2NetLossV3(nn.Module):
    def __init__(self):
        super(U2NetLossV3, self).__init__()

        self.bce_loss = nn.BCELoss(size_average=True)
        self.boundary_loss = BoundaryLoss()

    def forward(self, pred, batch, alpha=0.5):
        d0 = pred["d0"]
        d1 = pred["d1"]
        d2 = pred["d2"]
        d3 = pred["d3"]
        d4 = pred["d4"]
        d5 = pred["d5"]
        d6 = pred["d6"]

        label = batch["label"]
        dist_map = batch["dist_map"]

        d0_bce = self.bce_loss(d0, label)
        d1_bce = self.bce_loss(d1, label)
        d2_bce = self.bce_loss(d2, label)
        d3_bce = self.bce_loss(d3, label)
        d4_bce = self.bce_loss(d4, label)
        d5_bce = self.bce_loss(d5, label)
        d6_bce = self.bce_loss(d6, label)

        bce_loss = d0_bce + d1_bce + d2_bce + d3_bce + d4_bce + d5_bce + d6_bce

        d0_boundary = self.boundary_loss(d0, dist_map)
        d1_boundary = self.boundary_loss(d1, dist_map)
        d2_boundary = self.boundary_loss(d2, dist_map)
        d3_boundary = self.boundary_loss(d3, dist_map)
        d4_boundary = self.boundary_loss(d4, dist_map)
        d5_boundary = self.boundary_loss(d5, dist_map)
        d6_boundary = self.boundary_loss(d6, dist_map)

        boundary_loss = d0_boundary + d1_boundary + d2_boundary + d3_boundary + d4_boundary + d5_boundary + d6_boundary

        loss = bce_loss + boundary_loss

        metrics = OrderedDict(
            d0_bce=d0_bce, d1_bce=d1_bce, d2_bce=d2_bce, d3_bce=d3_bce,
            d4_bce=d4_bce, d5_bce=d5_bce, d6_bce=d6_bce, bce_loss=bce_loss,
            d0_boundary=d0_boundary, d1_boundary=d1_boundary, d2_boundary=d2_boundary, d3_boundary=d3_boundary,
            d4_boundary=d4_boundary, d5_boundary=d5_boundary, d6_boundary=d6_boundary, boundary_loss=boundary_loss,
            loss=loss
        )

        return loss, metrics
