# octscreen/explain/__init__.py
"""Explainer factory (contract C6). The app calls only build_explainer() and Explainer.explain().
If torch/timm aren't installed or the package has no explainer, build_explainer() returns None and the app hides heatmaps."""
import warnings
from pathlib import Path


class Explainer:
    def explain(self, pil_image, class_idx):
        """Return a heatmap (H, W) float32 in [0, 1] at the model's input size."""
        raise NotImplementedError


class GradCAMExplainer(Explainer):
    def __init__(self, checkpoint_path, target_layer="auto"):
        import timm
        import torch
        from octscreen.explain.gradcam import GradCAM
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)   # our own trusted file
        model = timm.create_model(ck["arch"], pretrained=False, num_classes=ck["num_classes"])
        model.load_state_dict(ck["state_dict"])
        self._torch = torch
        self.spec = ck["preprocess"]
        self.class_names = list(ck["class_names"])
        self.cam = GradCAM(model, target_layer, tuple(self.spec["image_size"]))

    def explain(self, pil_image, class_idx):
        from octscreen.common.preprocess import preprocess
        return self.cam(self._torch.from_numpy(preprocess(pil_image, self.spec)), class_idx)


def build_explainer(meta, package_dir):
    ex = meta.get("explainer")
    if not ex:
        return None
    try:
        if ex["type"] == "gradcam_torch":
            return GradCAMExplainer(Path(package_dir) / ex["file"], ex.get("target_layer", "auto"))
        warnings.warn(f"Unknown explainer type {ex['type']!r}: heatmaps disabled")
    except ImportError as e:
        warnings.warn(f"Explainer needs torch + timm ({e}): heatmaps disabled")
    return None