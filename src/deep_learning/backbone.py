import torch
from torch import nn

def build_backbone(arch="tf_efficientnetv2_rw_s", pretrained=True):
    """Uses timm architecture name. Set pretrained=False when loading a task checkpoint."""
    import timm
    model=timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool="avg")
    return model, model.num_features

def load_checkpoint(model, checkpoint_path, device="cpu"):
    """Strict load by default. Resolve classifier/head differences explicitly; do not silently ignore mismatches."""
    ckpt=torch.load(checkpoint_path, map_location=device, weights_only=False)
    state=ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    state={k.removeprefix("module."):v for k,v in state.items()}
    result=model.load_state_dict(state, strict=True)
    return result
