import torch
import torch.nn as nn
import torch.nn.functional as F

from addict import Dict
from collections import OrderedDict

from pseg.layers.u2net import RSU4F, RSU5, RSU6, RSU7, RSU4
from pseg.layers.common import DFMAtt, SEBlock


class U2NetHead(nn.Module):
    def __init__(self, config=None):
        super(U2NetHead, self).__init__()

        self.cfg = Dict(config)

        out_ch = self.cfg.get("out_ch", 3)
        self.model_name = self.cfg.get("model_name", "small")

        if self.model_name == "small":
            cfg = [
                [128, 16, 64],
                [128, 16, 64],
                [128, 16, 64],
                [128, 16, 64],
                [128, 16, 64]
            ]
        else:
            cfg = [
                [1024, 256, 512],
                [1024, 128, 256],
                [512, 64, 128],
                [256, 32, 64],
                [128, 16, 64],
            ]

        # decoder
        self.stage5d = RSU4F(cfg[0][0], cfg[0][1], cfg[0][2])
        self.stage4d = RSU4(cfg[1][0], cfg[1][1], cfg[1][2])
        self.stage3d = RSU5(cfg[2][0], cfg[2][1], cfg[2][2])
        self.stage2d = RSU6(cfg[3][0], cfg[3][1], cfg[3][2])
        self.stage1d = RSU7(cfg[4][0], cfg[4][1], cfg[4][2])

        self.side1 = nn.Conv2d(cfg[4][2], out_ch, 3, padding=1)  # 16
        self.side2 = nn.Conv2d(cfg[3][2], out_ch, 3, padding=1)  # 32
        self.side3 = nn.Conv2d(cfg[2][2], out_ch, 3, padding=1)  # 64
        self.side4 = nn.Conv2d(cfg[1][2], out_ch, 3, padding=1)  # 128
        self.side5 = nn.Conv2d(cfg[0][2], out_ch, 3, padding=1)  # 512
        self.side6 = nn.Conv2d(cfg[0][2], out_ch, 3, padding=1)  # 512

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, features):
        hx1, hx2, hx3, hx4, hx5, hx6 = features
        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode="bilinear", align_corners=False)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode="bilinear", align_corners=False)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode="bilinear", align_corners=False)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=False)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=False)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=d1.shape[2:], mode="bilinear", align_corners=False)

        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=d1.shape[2:], mode="bilinear", align_corners=False)

        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=d1.shape[2:], mode="bilinear", align_corners=False)

        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=d1.shape[2:], mode="bilinear", align_corners=False)

        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, size=d1.shape[2:], mode="bilinear", align_corners=False)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        result = OrderedDict(
            d0=torch.sigmoid(d0), d1=torch.sigmoid(d1),
            d2=torch.sigmoid(d2), d3=torch.sigmoid(d3),
            d4=torch.sigmoid(d4), d5=torch.sigmoid(d5),
            d6=torch.sigmoid(d6)
        )

        return result


class U2NetHeadV1(nn.Module):
    def __init__(self, config=None):
        super(U2NetHeadV1, self).__init__()

        self.cfg = Dict(config)

        out_ch = self.cfg.get("out_ch", 1)
        self.model_name = self.cfg.get("model_name", "small")

        if self.model_name == "small":
            cfg = [
                [128, 16, 64],
                [128, 16, 64],
                [128, 16, 64],
                [128, 16, 64],
                [128, 16, 64]
            ]
        else:
            cfg = [
                [1024, 256, 512],
                [1024, 128, 256],
                [512, 64, 128],
                [256, 32, 64],
                [128, 16, 64],
            ]

        # decoder
        self.stage5d = RSU4F(cfg[0][0] + 1, cfg[0][1], cfg[0][2])
        self.stage4d = RSU4(cfg[1][0] + 1, cfg[1][1], cfg[1][2])
        self.stage3d = RSU5(cfg[2][0] + 1, cfg[2][1], cfg[2][2])
        self.stage2d = RSU6(cfg[3][0] + 1, cfg[3][1], cfg[3][2])
        self.stage1d = RSU7(cfg[4][0] + 1, cfg[4][1], cfg[4][2])

        self.side1 = nn.Conv2d(cfg[4][2], 4 * out_ch, 3, padding=1)  # 16
        self.side2 = nn.Conv2d(cfg[3][2], 4 * out_ch, 3, padding=1)  # 32
        self.side3 = nn.Conv2d(cfg[2][2], 4 * out_ch, 3, padding=1)  # 64
        self.side4 = nn.Conv2d(cfg[1][2], 4 * out_ch, 3, padding=1)  # 128
        self.side5 = nn.Conv2d(cfg[0][2], 4 * out_ch, 3, padding=1)  # 512
        self.side6 = nn.Conv2d(cfg[0][2], 4 * out_ch, 3, padding=1)  # 512

        self.ps1 = nn.PixelShuffle(2)
        self.ps2 = nn.PixelShuffle(2)
        self.ps3 = nn.PixelShuffle(2)
        self.ps4 = nn.PixelShuffle(2)
        self.ps5 = nn.PixelShuffle(2)
        self.ps6 = nn.PixelShuffle(2)

    def forward(self, features):
        hx1, hx2, hx3, hx4, hx5, hx6 = features

        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode="bilinear", align_corners=False)

        # side 6
        d6 = self.side6(hx6)
        d6 = self.ps6(d6)

        hx5d = self.stage5d(torch.cat((d6, hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode="bilinear", align_corners=False)

        # side 5
        d5 = self.side5(hx5d)
        d5 = self.ps5(d5)

        hx4d = self.stage4d(torch.cat((d5, hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode="bilinear", align_corners=False)

        # side 4
        d4 = self.side4(hx4d)
        d4 = self.ps4(d4)

        hx3d = self.stage3d(torch.cat((d4, hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=False)

        d3 = self.side3(hx3d)
        d3 = self.ps3(d3)

        hx2d = self.stage2d(torch.cat((d3, hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=False)

        d2 = self.side2(hx2d)
        d2 = self.ps2(d2)

        hx1d = self.stage1d(torch.cat((d2, hx2dup, hx1), 1))

        # side 1
        d1 = self.side1(hx1d)
        d1 = self.ps1(d1)

        d2 = F.interpolate(d2, size=d1.shape[2:], mode="bilinear", align_corners=False)

        d3 = F.interpolate(d3, size=d1.shape[2:], mode="bilinear", align_corners=False)

        d4 = F.interpolate(d4, size=d1.shape[2:], mode="bilinear", align_corners=False)

        d5 = F.interpolate(d5, size=d1.shape[2:], mode="bilinear", align_corners=False)

        d6 = F.interpolate(d6, size=d1.shape[2:], mode="bilinear", align_corners=False)

        # d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        # print(d1.shape, d2.shape, d3.shape, d4.shape, d5.shape, d6.shape)

        result = OrderedDict(
            # d0=torch.sigmoid(d0),
            d1=torch.sigmoid(d1),
            d2=torch.sigmoid(d2),
            d3=torch.sigmoid(d3),
            d4=torch.sigmoid(d4),
            d5=torch.sigmoid(d5),
            d6=torch.sigmoid(d6)
        )

        return result


class U2NetHeadV2(nn.Module):
    def __init__(self, config=None):
        super(U2NetHeadV2, self).__init__()

        self.cfg = Dict(config)

        out_ch = self.cfg.get("out_ch", 1)
        self.model_name = self.cfg.get("model_name", "small")

        if self.model_name == "small":
            cfg = [
                [80, 16, 64],
                [80, 16, 64],
                [80, 16, 64],
                [80, 16, 64]
            ]
        else:
            cfg = [
                [768, 128, 512],
                [384, 64, 256],
                [192, 32, 128],
                [96, 16, 64],
            ]

            # cfg = [
            #     [1024, 16, 64],
            #     [512, 16, 64],
            #     [256, 16, 64],
            #     [128, 16, 64],
            # ]

        # decoder
        self.stage4d = RSU4(cfg[0][0], cfg[0][1], cfg[0][2])
        self.stage3d = RSU5(cfg[1][0], cfg[1][1], cfg[1][2])
        self.stage2d = RSU6(cfg[2][0], cfg[2][1], cfg[2][2])
        self.stage1d = RSU7(cfg[3][0], cfg[3][1], cfg[3][2])

        # side
        self.side1 = nn.Conv2d(cfg[3][2], 4 * out_ch, 1)  # 64
        self.side2 = nn.Conv2d(cfg[2][2], 4 * out_ch, 1)  # 128
        self.side3 = nn.Conv2d(cfg[1][2], 4 * out_ch, 1)  # 256
        self.side4 = nn.Conv2d(cfg[0][2], 4 * out_ch, 1)  # 512

        # self.side1 = nn.Sequential(
        #     nn.Conv2d(cfg[3][2], cfg[3][2], kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(cfg[3][2]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(cfg[3][2], 4 * out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        # )
        # self.side2 = nn.Sequential(
        #     nn.Conv2d(cfg[2][2], cfg[2][2], kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(cfg[2][2]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(cfg[2][2], 4 * out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        # )
        # self.side3 = nn.Sequential(
        #     nn.Conv2d(cfg[1][2], cfg[1][2], kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(cfg[1][2]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(cfg[1][2], 4 * out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        # )
        # self.side4 = nn.Sequential(
        #     nn.Conv2d(cfg[0][2], cfg[0][2], kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(cfg[0][2]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(cfg[0][2], 4 * out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        # )

        # pixel shuffle
        self.ps1 = nn.PixelShuffle(2)
        self.ps2 = nn.PixelShuffle(2)
        self.ps3 = nn.PixelShuffle(2)
        self.ps4 = nn.PixelShuffle(2)
        self.ps5 = nn.PixelShuffle(2)

        # attention
        self.attn1 = DFMAtt(cfg[3][2], cfg[3][2], k=1)
        self.attn2 = DFMAtt(cfg[2][2], cfg[2][2], k=1)
        self.attn3 = DFMAtt(cfg[1][2], cfg[1][2], k=1)
        self.attn4 = DFMAtt(cfg[0][2], cfg[0][2], k=1)
        self.attn5 = DFMAtt(cfg[0][2] * 2, cfg[0][2] * 2, k=1)

        self.outconv = nn.Conv2d(4, out_ch, 1)

    def forward(self, features):
        hx1, hx2, hx3, hx4, hx5 = features

        hx5 = self.attn5(hx5)
        hx5up = self.ps5(hx5)

        # stage4d
        hx4d = self.stage4d(torch.cat((hx5up, hx4), 1))
        hx4d = self.attn4(hx4d)
        hx4dup = self.ps4(hx4d)

        # side 4
        d4 = self.side4(hx4d)
        d4 = self.ps4(d4)

        # stage3d
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3d = self.attn3(hx3d)
        hx3dup = self.ps3(hx3d)

        # side 3
        d3 = self.side3(hx3d)
        d3 = self.ps3(d3)

        # stage2d
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2d = self.attn2(hx2d)
        hx2dup = self.ps2(hx2d)

        # side 2
        d2 = self.side2(hx2d)
        d2 = self.ps2(d2)

        # stage1d
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.attn1(hx1d)

        # side 1
        d1 = self.side1(hx1d)
        d1 = self.ps1(d1)

        # interpolate
        d2 = F.interpolate(d2, size=d1.shape[2:], mode="bilinear", align_corners=False)
        d3 = F.interpolate(d3, size=d1.shape[2:], mode="bilinear", align_corners=False)
        d4 = F.interpolate(d4, size=d1.shape[2:], mode="bilinear", align_corners=False)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4), 1))

        result = OrderedDict(
            d0=torch.sigmoid(d0),
            d1=torch.sigmoid(d1),
            d2=torch.sigmoid(d2),
            d3=torch.sigmoid(d3),
            d4=torch.sigmoid(d4)
        )

        return result


class U2NetHeadV3(nn.Module):
    def __init__(self, config=None):
        super(U2NetHeadV3, self).__init__()

        self.cfg = Dict(config)

        out_ch = self.cfg.get("out_ch", 1)
        self.model_name = self.cfg.get("model_name", "small")

        if self.model_name == "small":
            cfg = [
                [80, 16, 64],
                [80, 16, 64],
                [80, 16, 64],
                [80, 16, 64]
            ]
        else:
            cfg = [
                [32, 16, 32],
                [64, 32, 64],
                [128, 64, 128],
                [256, 128, 256],
                [512, 512, 512]
            ]

        # decoder
        self.stage1d = RSU7(cfg[0][0], cfg[0][1], cfg[0][2])
        self.stage2d = RSU6(cfg[1][0], cfg[1][1], cfg[1][2])
        self.stage3d = RSU5(cfg[2][0], cfg[2][1], cfg[2][2])
        self.stage4d = RSU4(cfg[3][0], cfg[3][1], cfg[3][2])

        # compress
        self.c1 = nn.Conv2d(cfg[0][2] * 2, cfg[0][1], 1)
        self.c2 = nn.Conv2d(cfg[1][2] * 2, cfg[1][1], 1)
        self.c3 = nn.Conv2d(cfg[2][2] * 2, cfg[2][1], 1)
        self.c4 = nn.Conv2d(cfg[3][2] * 2, cfg[3][1], 1)

        self.se5 = SEBlock(cfg[4][2], cfg[4][1])

        # side
        self.side1 = nn.Conv2d(cfg[0][2], 4 * out_ch, 1)  # 64
        self.side2 = nn.Conv2d(cfg[1][2], 4 * out_ch, 1)  # 128
        self.side3 = nn.Conv2d(cfg[2][2], 4 * out_ch, 1)  # 256
        self.side4 = nn.Conv2d(cfg[3][2], 4 * out_ch, 1)  # 512

        # pixel shuffle
        self.ps0 = nn.PixelShuffle(2)
        self.ps1 = nn.PixelShuffle(2)
        self.ps2 = nn.PixelShuffle(2)
        self.ps3 = nn.PixelShuffle(2)
        self.ps4 = nn.PixelShuffle(2)
        self.ps5 = nn.PixelShuffle(2)

        # attention
        self.attn1 = DFMAtt(cfg[0][1], cfg[0][1], k=4)
        self.attn2 = DFMAtt(cfg[1][1], cfg[1][1], k=4)
        self.attn3 = DFMAtt(cfg[2][1], cfg[2][1], k=4)
        self.attn4 = DFMAtt(cfg[3][1], cfg[3][1], k=4)

        # self.outconv = nn.Conv2d(16, out_ch * 4, 1)

    def forward(self, features):
        hx1, hx2, hx3, hx4, hx5 = features

        # compress
        hx1 = self.c1(hx1)
        hx2 = self.c2(hx2)
        hx3 = self.c3(hx3)
        hx4 = self.c4(hx4)

        # attention
        hx1 = self.attn1(hx1)
        hx2 = self.attn2(hx2)
        hx3 = self.attn3(hx3)
        hx4 = self.attn4(hx4)

        # se block
        hx5 = self.se5(hx5)

        hx5up = self.ps5(hx5)

        # stage4d
        hx4d = self.stage4d(torch.cat((hx5up, hx4), 1))
        hx4dup = self.ps4(hx4d)

        # side 4
        d4 = self.side4(hx4d)
        d4ps = self.ps4(d4)

        # stage3d
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self.ps3(hx3d)

        # side 3
        d3 = self.side3(hx3d)
        d3ps = self.ps3(d3)

        # stage2d
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.ps2(hx2d)

        # side 2
        d2 = self.side2(hx2d)
        d2ps = self.ps2(d2)

        # stage1d
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side 1
        d1 = self.side1(hx1d)
        d1ps = self.ps1(d1)

        # interpolate
        # d2 = F.interpolate(d2, size=d1.shape[2:], mode="bilinear", align_corners=False)
        # d3 = F.interpolate(d3, size=d1.shape[2:], mode="bilinear", align_corners=False)
        # d4 = F.interpolate(d4, size=d1.shape[2:], mode="bilinear", align_corners=False)

        d2ps = F.interpolate(d2ps, size=d1ps.shape[2:], mode="bilinear", align_corners=False)
        d3ps = F.interpolate(d3ps, size=d1ps.shape[2:], mode="bilinear", align_corners=False)
        d4ps = F.interpolate(d4ps, size=d1ps.shape[2:], mode="bilinear", align_corners=False)
        # print(d1ps.shape, d2ps.shape, d3ps.shape, d4ps.shape)

        # d0 = self.outconv(torch.cat((d1, d2, d3, d4), 1))
        # d0ps = self.ps0(d0)

        result = OrderedDict(
            # d0=torch.sigmoid(d0ps),
            d1=torch.sigmoid(d1ps),
            d2=torch.sigmoid(d2ps),
            d3=torch.sigmoid(d3ps),
            d4=torch.sigmoid(d4ps)
        )

        return result
