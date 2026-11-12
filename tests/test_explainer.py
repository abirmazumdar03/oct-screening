# tests/test_explainer.py
import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")


def test_gradcam_shape_and_range():
    from octscreen.explain.gradcam import GradCAM
    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=4).eval()
    cam = GradCAM(model, "auto", input_size=(64, 64))
    h = cam(torch.randn(1, 3, 64, 64), 0)
    assert h.shape == (64, 64) and h.dtype == np.float32
    assert 0.0 <= float(h.min()) and float(h.max()) <= 1.0


def test_overlay_keeps_original_size():
    from octscreen.explain.overlay import overlay
    img = Image.new("L", (512, 496), 100)
    out = overlay(img, np.random.default_rng(0).random((224, 224)).astype(np.float32))
    assert out.size == (512, 496) and out.mode == "RGB"