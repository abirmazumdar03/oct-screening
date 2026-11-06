# tests/test_app_safety.py
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from octscreen.app.safety import check_image

G = OmegaConf.create({"min_side_px": 128, "aspect_min": 0.5, "aspect_max": 4.0, "max_color_diff": 12})


def codes(img):
    return [c for c, _ in check_image(img, G)]


def test_grayscale_oct_like_image_passes():
    arr = (np.random.default_rng(0).random((496, 768)) * 255).astype(np.uint8)
    assert codes(Image.fromarray(arr)) == []


def test_colour_photo_is_rejected():
    arr = np.zeros((480, 640, 3), np.uint8)
    arr[..., 0], arr[..., 1] = 220, 120                         # orange-ish
    assert "not_grayscale" in codes(Image.fromarray(arr))


def test_tiny_and_odd_shapes_are_rejected():
    assert "too_small" in codes(Image.new("L", (64, 64)))
    assert "bad_aspect" in codes(Image.new("L", (2000, 200)))


def test_real_sample_scans_pass_if_available():
    folder = Path("artifacts/packs/sample_scans")
    files = [p for p in folder.rglob("*") if p.suffix.lower() in (".jpeg", ".jpg", ".png")] if folder.exists() else []
    if not files:
        pytest.skip("sample_scans not unpacked")
    rejected = [p.name for p in files if codes(Image.open(p))]
    assert rejected == [], f"real OCT scans rejected by the guard: {rejected[:5]}"
