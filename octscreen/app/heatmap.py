# octscreen/app/heatmap.py
"""Built-in Grad-CAM fallback for the app, used while octscreen/explain (contract C6, Abir) is still empty.
As soon as octscreen.explain provides build_explainer + overlay, inference.py uses those instead.

Reads the package's explainer exactly as octscreen/optimize/package.py writes it:
  model_meta.json → "explainer": {"type": "gradcam_torch", "file": "explainer.pt", "arch": ..., "target_layer": "auto"}  (or null)
  explainer.pt    → {"format_version", "state_dict", "arch", "num_classes", "class_names", "preprocess"}
Prediction never depends on this file: no PyTorch → no heatmap, everything else works.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from octscreen.common.preprocess import preprocess


def build_explainer(meta, package_dir):
    ex = meta.get("explainer")
    if not ex:
        print("[heatmap] package has no explainer → heatmaps off")
        return None
    if ex.get("type", "gradcam_torch") != "gradcam_torch":
        print(f"[heatmap] explainer type {ex.get('type')!r} not supported → heatmaps off")
        return None
    path = Path(str(package_dir)) / ex["file"]
    try:
        import timm
        import torch
    except ImportError:
        print("[heatmap] PyTorch/timm not installed (group app-heatmap) → heatmaps off")
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)        # our own package file
        if isinstance(ck, torch.nn.Module):
            model = ck
        else:
            arch = ck.get("arch") or ex.get("arch")
            num_classes = int(ck.get("num_classes") or len(meta["class_names"]))
            model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
            model.load_state_dict(ck["state_dict"])
        print(f"[heatmap] Grad-CAM ready ({ex.get('arch') or type(model).__name__})")
        return GradCAM(model.eval(), meta["preprocess"])
    except Exception as e:                                                    # never let heatmaps stop the app
        print(f"[heatmap] could not load {path.name}: {e} → heatmaps off")
        return None


class GradCAM:
    """Grad-CAM on the last feature map of a timm CNN (forward_features → forward_head)."""

    def __init__(self, model, preprocess_spec):
        self.model, self.spec = model, preprocess_spec

    def explain(self, img, class_idx):
        import torch
        x = torch.from_numpy(preprocess(img, self.spec))
        with torch.enable_grad():
            feats = self.model.forward_features(x)
            if feats.ndim != 4:
                raise ValueError(f"Grad-CAM needs a [1, C, h, w] feature map, got {tuple(feats.shape)}")
            logits = self.model.forward_head(feats)
            grads = torch.autograd.grad(logits[0, int(class_idx)], feats)[0]
        cam = torch.relu((grads.mean(dim=(2, 3), keepdim=True) * feats).sum(dim=1))[0].detach().numpy()
        cam = cam - cam.min()
        return cam / cam.max() if cam.max() > 0 else cam


def _jet(v):
    v = np.clip(v, 0.0, 1.0)
    r, g, b = (np.clip(1.5 - np.abs(4 * v - k), 0, 1) for k in (3, 2, 1))
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(img, cam, alpha=0.45):
    """Blend a [0, 1] heatmap over the (grayscale) scan. NumPy + Pillow only."""
    base = img.convert("L").convert("RGB")
    heat = Image.fromarray((np.clip(np.asarray(cam, np.float32), 0, 1) * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR)
    out = (1 - float(alpha)) * np.asarray(base, np.float32) + float(alpha) * _jet(np.asarray(heat, np.float32) / 255.0)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def explain_api():
    """(build_explainer, overlay): Abir's octscreen.explain if it's filled in, else this module's."""
    try:
        from octscreen.explain import build_explainer as abir_build
        from octscreen.explain.overlay import overlay as abir_overlay
        return abir_build, abir_overlay
    except ImportError:
        return build_explainer, overlay
