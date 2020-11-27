import torch.nn as nn

from addict import Dict

from pseg.modeling.backbone.u2net_backbone import U2NetBackbone, U2NetBackboneV1
from pseg.modeling.head.u2net_head import U2NetHead, U2NetHeadV1


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


if __name__ == "__main__":
    import torch

    model = U2Net({
        "backbone": {"type": "U2NetBackboneV1", "args": {"in_ch": 3, "model_name": "large"}},
        "head": {"type": "U2NetHeadV1", "args": {"out_ch": 1, "model_name": "large"}}
    })

    model(torch.ones(1, 3, 640, 640))
