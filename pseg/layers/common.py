import torch
import torch.nn as nn
import torch.nn.functional as F

from pseg.layers.activation import HSigmoid


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat(
            [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]],
            1
        )


class DFMAtt(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super(DFMAtt, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        self.k = k
        self.out_ch = out_ch
        offset_list = []
        for x in range(k):
            conv = nn.Conv2d(in_ch, 2, 1, 1, 0, bias=True)
            offset_list.append(conv)
        self.offset_conv = nn.ModuleList(offset_list)
        self.weight_conv = nn.Sequential(nn.Conv2d(in_ch, k, 1, 1, 0, bias=True), nn.Softmax(1))

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size()
        proj_feat = self.conv(x)
        offsets = []
        for i in range(self.k):
            flow = self.offset_conv[i](x)
            offsets.append(flow)
        offset_weights = torch.repeat_interleave(self.weight_conv(x), self.out_ch, 1)
        feats = []
        for i in range(self.k):
            flow = offsets[i]
            flow = flow.permute(0, 2, 3, 1)
            grid_y, grid_x = torch.meshgrid([torch.arange(0, h), torch.arange(0, w)])
            grid = torch.stack((grid_x, grid_y), 2).float()
            grid.requires_grad = False
            grid = grid.type_as(proj_feat)
            vgrid = grid + flow
            vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            feat = F.grid_sample(proj_feat, vgrid_scaled, mode="bilinear", padding_mode="zeros", align_corners=False)
            feats.append(feat)
        feat = torch.cat(feats, 1) * offset_weights
        feat = sum(torch.split(feat, self.out_ch, 1))
        return feat


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=4):
        super(SEBlock, self).__init__()

        num_mid_filter = out_channels // ratio
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_mid_filter, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_mid_filter, kernel_size=1, out_channels=out_channels, bias=True)
        self.relu2 = HSigmoid()

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv1(attn)
        attn = self.relu1(attn)
        attn = self.conv2(attn)
        attn = self.relu2(attn)
        return x * attn
