# octscreen/explain/gradcam.py
"""Grad-CAM (Selvaraju et al., 2017) with plain PyTorch hooks. Works for any timm CNN."""
import numpy as np
import torch
import torch.nn.functional as F


def find_target_layer(model, input_size):
    """Name of the LAST module (in execution order) that outputs a 4-D feature map with H, W > 1.
    For timm CNNs that's the end of the feature extractor (e.g. 'blocks' / 'layer4'), just before pooling."""
    calls = []
    hooks = [m.register_forward_hook(lambda mod, inp, out, n=name: calls.append((n, tuple(out.shape) if torch.is_tensor(out) else None)))
             for name, m in model.named_modules() if name]
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, *input_size))
    finally:
        for h in hooks:
            h.remove()
    for name, shape in reversed(calls):
        if shape and len(shape) == 4 and shape[-1] > 1 and shape[-2] > 1:
            return name
    raise RuntimeError("No 4-D feature map found; set package.target_layer to a module name explicitly")


class GradCAM:
    def __init__(self, model, target_layer="auto", input_size=(224, 224)):
        self.model = model.eval()
        self.layer_name = find_target_layer(model, tuple(input_size)) if target_layer in (None, "auto") else str(target_layer)
        modules = dict(model.named_modules())
        if self.layer_name not in modules:
            raise KeyError(f"target_layer {self.layer_name!r} not in model")
        self._acts = self._grads = None
        modules[self.layer_name].register_forward_hook(self._on_forward)

    def _on_forward(self, module, inp, out):
        self._acts = out
        if out.requires_grad:
            out.register_hook(self._on_backward)

    def _on_backward(self, grad):
        self._grads = grad

    def __call__(self, x, class_idx):
        """x: (1, 3, H, W) float tensor. Returns heatmap (H, W) float32 in [0, 1]."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            logits[0, int(class_idx)].backward()
        if self._grads is None:
            raise RuntimeError("No gradients captured: is the explainer model's requires_grad disabled?")
        weights = self._grads.mean(dim=(2, 3), keepdim=True)                 # importance of each channel
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))        # weighted sum, positive part only
        cam = F.interpolate(cam, size=tuple(x.shape[-2:]), mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        peak = cam.max()
        cam = cam / peak if peak > 0 else cam
        self._acts = self._grads = None
        return cam.detach().cpu().numpy().astype(np.float32)