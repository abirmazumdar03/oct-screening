# tests/test_app_safety.py
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from octscreen.app.safety import check_image

G = OmegaConf.create({"min_side_px": 128, "aspect_min": 0.5, "aspect_max": 4.0, "max_color_diff": 4,
                      "color_pixel_chroma": 20, "max_color_fraction": 0.02})


def codes(img):
    return [c for c, _ in check_image(img, G)]


def test_grayscale_oct_like_image_passes():
    arr = (np.random.default_rng(0).random((496, 768)) * 255).astype(np.uint8)
    assert codes(Image.fromarray(arr)) == []


def test_colour_photo_is_rejected():
    arr = np.zeros((480, 640, 3), np.uint8)
    arr[..., 0], arr[..., 1] = 220, 120                         # orange-ish
    assert "not_grayscale" in codes(Image.fromarray(arr))


def test_muted_indoor_photo_is_rejected():
    """Regression (14 Nov): a selfie against grey walls averaged only 9.3 and was screened as 'CNV 100%'."""
    arr = np.full((480, 640, 3), 150, np.uint8)                 # grey wall
    arr[120:360, 160:480] = (205, 160, 130)                     # skin/shirt region, 25% of the image
    assert "not_grayscale" in codes(Image.fromarray(arr))


def test_grayscale_scan_with_slight_noise_passes():
    base = (np.random.default_rng(1).random((496, 768)) * 255).astype(np.int16)
    rgb = np.stack([base, base + 1, base], axis=-1).clip(0, 255).astype(np.uint8)   # JPEG-like ±1 tint
    assert codes(Image.fromarray(rgb)) == []


def test_app_yaml_guard_settings_reject_the_muted_photo():
    from octscreen.common.config import load_config
    g = load_config("configs/app.yaml").policy.guard
    arr = np.full((480, 640, 3), 150, np.uint8)
    arr[120:360, 160:480] = (205, 160, 130)
    assert "not_grayscale" in [c for c, _ in check_image(Image.fromarray(arr), g)]


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


# ---- structure check ("does it look like an OCT B-scan?") ------------------------------------------------
GS = OmegaConf.merge(G, {"structure": {"enabled": True, "min_row_peak": 0.689,
                                       "max_median_brightness": 86, "min_speckle": 0.068}})


def _oct_like(seed=0):
    """Dark background, one bright horizontal band, grainy speckle: the basic look of an OCT B-scan."""
    rng = np.random.default_rng(seed)
    a = np.full((496, 768), 15.0)
    a[220:280] = 170
    a += rng.normal(0, 18, a.shape)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def test_structure_passes_oct_like_image():
    assert [c for c, _ in check_image(_oct_like(), GS)] == []


def test_structure_rejects_smooth_xray_like_image():
    y, x = np.mgrid[0:600, 0:500]
    blob = 60 + 150 * np.exp(-(((x - 250) / 140) ** 2 + ((y - 300) / 220) ** 2))     # smooth, bright, no band
    assert "not_oct_structure" in [c for c, _ in check_image(Image.fromarray(blob.astype(np.uint8)), GS)]


def test_structure_check_is_optional():
    """Without a `structure` block the guard behaves exactly as before (lego: nothing breaks)."""
    y, x = np.mgrid[0:600, 0:500]
    blob = 60 + 150 * np.exp(-(((x - 250) / 140) ** 2 + ((y - 300) / 220) ** 2))
    assert "not_oct_structure" not in codes(Image.fromarray(blob.astype(np.uint8)))


def test_app_yaml_structure_keeps_most_real_scans_if_available():
    """Measured cost: ~2% of real scans are rejected. Guard against a config change that rejects far more."""
    from octscreen.common.config import load_config
    g = load_config("configs/app.yaml").policy.guard
    folder = Path("artifacts/packs/sample_scans")
    files = [p for p in folder.rglob("*") if p.suffix.lower() in (".jpeg", ".jpg", ".png")] if folder.exists() else []
    if not files:
        pytest.skip("sample_scans not unpacked")
    rejected = [p.name for p in files if check_image(Image.open(p), g)]
    assert len(rejected) <= max(1, int(0.05 * len(files))), f"too many real scans rejected: {rejected}"
