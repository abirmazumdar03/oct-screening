# octscreen/explain/overlay.py
import numpy as np
from PIL import Image


def colormap_jet(h):
    """[0,1] array → RGB uint8 (blue = low, red = high)."""
    h = np.clip(h, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(pil_image, heatmap, alpha=0.45, size=None):
    """Blend a heatmap onto the scan. Output is at the original image size (or `size`).
    Exact for resize_mode='squash' (the heatmap is stretched back the same way the image was squashed)."""
    base = pil_image.convert("RGB")
    if size:
        base = base.resize(tuple(size), Image.BILINEAR)
    heat = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR)
    colored = Image.fromarray(colormap_jet(np.asarray(heat, dtype=np.float32) / 255.0))
    return Image.blend(base, colored, float(alpha))