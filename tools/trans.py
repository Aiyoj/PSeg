if __name__ == "__main__":
    import re
    import torch
    from collections import OrderedDict

    from pseg.algorithm.u2net import U2Net

    model = U2Net({"model_name": "large"})
    print(model)

    new_model_weight = OrderedDict()
    model_weight = torch.load("models/ori_u2net.pth", map_location=torch.device("cpu"))
    for k, v in model_weight.items():
        if len(re.findall("stage1|stage2|stage3|stage4|stage5|stage6", k)) > 0 and len(
                re.findall("stage1d|stage2d|stage3d|stage4d|stage5d", k)) == 0:
            new_model_weight[f"backbone.{k}"] = v
            # print(k)

        if len(re.findall("stage1d|stage2d|stage3d|stage4d|stage5d", k)) > 0:
            new_model_weight[f"head.{k}"] = v
            # print(k)

        if len(re.findall("side1|side2|side3|side4|side5|side6", k)) > 0:
            new_model_weight[f"head.{k}"] = v
            # print(k)

        if len(re.findall("outconv\.weight|outconv\.bias", k)) > 0:
            new_model_weight[f"head.{k}"] = v
            # print(k)

    model.load_state_dict(new_model_weight, strict=True)
    model.eval()
    torch.save(model.state_dict(), "models/u2net.pth")
