import torch.nn as nn

from addict import Dict

from pseg.modeling.backbone.u2net_backbone import U2NetBackbone, U2NetBackboneV1, U2NetBackboneV2, U2NetBackboneV3
from pseg.modeling.head.u2net_head import U2NetHead, U2NetHeadV1, U2NetHeadV2, U2NetHeadV3


class U2Net(nn.Module):
    def __init__(self, config=None):
        super(U2Net, self).__init__()

        self.cfg = Dict(config)

        self.backbone_type = self.cfg.backbone.get("type", "U2NetBackbone")
        self.backbone_args = self.cfg.backbone.get(
            "args", {"in_ch": 3, "model_name": "large"}
        )
        self.head_type = self.cfg.head.get("type", "U2NetHead")
        self.head_args = self.cfg.head.get(
            "args", {"out_ch": 1, "model_name": "large"}
        )

        self.backbone = eval(self.backbone_type)(self.backbone_args)
        self.head = eval(self.head_type)(self.head_args)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
