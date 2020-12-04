import torch.nn as nn

from addict import Dict

from pseg.layers.common import Focus
from pseg.layers.u2net import RSU4F, RSU4, RSU5, RSU6, RSU7


class U2NetBackbone(nn.Module):
    def __init__(self, config=None):
        super(U2NetBackbone, self).__init__()

        self.cfg = Dict(config)
        in_ch = self.cfg.get("in_ch", 3)
        self.model_name = self.cfg.get("model_name", "small")

        if self.model_name == "small":
            cfg = [
                [in_ch, 16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [64, 16, 64]
            ]
        else:
            cfg = [
                [in_ch, 32, 64],
                [64, 32, 128],
                [128, 64, 256],
                [256, 128, 512],
                [512, 256, 512],
                [512, 256, 512]
            ]

        self.stage1 = RSU7(cfg[0][0], cfg[0][1], cfg[0][2])
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(cfg[1][0], cfg[1][1], cfg[1][2])
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(cfg[2][0], cfg[2][1], cfg[2][2])
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(cfg[3][0], cfg[3][1], cfg[3][2])
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(cfg[4][0], cfg[4][1], cfg[4][2])
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(cfg[5][0], cfg[5][1], cfg[5][2])

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)

        return [hx1, hx2, hx3, hx4, hx5, hx6]


class U2NetBackboneV1(nn.Module):
    def __init__(self, config=None):
        super(U2NetBackboneV1, self).__init__()

        self.cfg = Dict(config)
        in_ch = self.cfg.get("in_ch", 3)
        self.model_name = self.cfg.get("model_name", "small")

        if self.model_name == "small":
            cfg = [
                [in_ch * 4, 16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [64, 16, 64]
            ]
        else:
            cfg = [
                [in_ch * 4, 32, 64],
                [64, 32, 128],
                [128, 64, 256],
                [256, 128, 512],
                [512, 256, 512],
                [512, 256, 512]
            ]

        self.focus = Focus()

        self.stage1 = RSU7(cfg[0][0], cfg[0][1], cfg[0][2])
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(cfg[1][0], cfg[1][1], cfg[1][2])
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(cfg[2][0], cfg[2][1], cfg[2][2])
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(cfg[3][0], cfg[3][1], cfg[3][2])
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(cfg[4][0], cfg[4][1], cfg[4][2])
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(cfg[5][0], cfg[5][1], cfg[5][2])

    def forward(self, x):
        # focus
        hx = self.focus(x)

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)

        return [hx1, hx2, hx3, hx4, hx5, hx6]


class U2NetBackboneV2(nn.Module):
    def __init__(self, config=None):
        super(U2NetBackboneV2, self).__init__()

        self.cfg = Dict(config)
        in_ch = self.cfg.get("in_ch", 3)
        self.model_name = self.cfg.get("model_name", "small")

        if self.model_name == "small":
            cfg = [
                [in_ch * 4, 16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [64, 16, 64]
            ]
        else:
            cfg = [
                [in_ch * 4, 16, 64],
                [64, 32, 128],
                [128, 64, 256],
                [256, 128, 512],
                [512, 256, 1024]
            ]

        self.focus = Focus()

        self.stage1 = RSU7(cfg[0][0], cfg[0][1], cfg[0][2])
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(cfg[1][0], cfg[1][1], cfg[1][2])
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(cfg[2][0], cfg[2][1], cfg[2][2])
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(cfg[3][0], cfg[3][1], cfg[3][2])
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(cfg[4][0], cfg[4][1], cfg[4][2])

    def forward(self, x):
        # focus
        hx = self.focus(x)

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)

        return [hx1, hx2, hx3, hx4, hx5]


class U2NetBackboneV3(nn.Module):
    def __init__(self, config=None):
        super(U2NetBackboneV3, self).__init__()

        self.cfg = Dict(config)
        in_ch = self.cfg.get("in_ch", 3)
        self.model_name = self.cfg.get("model_name", "small")

        if self.model_name == "small":
            cfg = [
                [in_ch * 4, 16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [64, 16, 64]
            ]
        else:
            cfg = [
                [in_ch * 4, 16, 64],
                [64, 32, 128],
                [128, 64, 256],
                [256, 128, 512],
                [512, 256, 512]
            ]

        self.focus = Focus()

        self.stage1 = RSU7(cfg[0][0], cfg[0][1], cfg[0][2])
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(cfg[1][0], cfg[1][1], cfg[1][2])
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(cfg[2][0], cfg[2][1], cfg[2][2])
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(cfg[3][0], cfg[3][1], cfg[3][2])
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(cfg[4][0], cfg[4][1], cfg[4][2])

    def forward(self, x):
        # focus
        hx = self.focus(x)

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)

        return [hx1, hx2, hx3, hx4, hx5]
