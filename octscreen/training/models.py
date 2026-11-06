# octscreen/training/models.py
import timm
import torch


def build_model(arch, num_classes, pretrained=True, drop_rate=0.0):
    """Any timm architecture name works: change it in the config, not here."""
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


def _head_ids(model):
    return {id(p) for p in model.get_classifier().parameters()}


def split_param_groups(model, lr_backbone, lr_head, weight_decay):
    """Two learning rates: gentle for the pre-trained body, faster for the new classifier head."""
    head_ids = _head_ids(model)
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    head = [p for p in model.parameters() if id(p) in head_ids]
    return [
        {"params": backbone, "lr": float(lr_backbone), "weight_decay": float(weight_decay), "name": "backbone"},
        {"params": head, "lr": float(lr_head), "weight_decay": float(weight_decay), "name": "head"},
    ]


def set_backbone_trainable(model, trainable):
    head_ids = _head_ids(model)
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


def load_checkpoint_model(path, device="cpu"):
    """Rebuild any of our checkpoints (contract C3) without knowing what's inside."""
    ck = torch.load(path, map_location="cpu", weights_only=False)   # our own trusted file
    model = timm.create_model(ck["arch"], pretrained=False, num_classes=ck["num_classes"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
