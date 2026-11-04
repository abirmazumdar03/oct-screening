import numpy as np
from PIL import Image

RESAMPLE = Image.BILINEAR

def _letterbox(img: Image.Image, w: int, h: int) -> Image.Image:
    scale = min(w / img.width, h / img.height)
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    canvas = Image.new("L", (w, h), 0)
    canvas.paste(img.resize((nw, nh), RESAMPLE), ((w - nw) // 2, (h - nh) // 2))
    return canvas

def resize_gray(img: Image.Image, spec: dict) -> Image.Image:
    """Step 1: grayscale + resize to the model input size. Returns a PIL 'L' image."""
    h, w = spec["image_size"]
    img = img.convert("L")
    mode = spec.get("resize_mode", "squash")
    if mode == "squash":
        return img.resize((w, h), RESAMPLE)
    if mode == "pad":
        return _letterbox(img, w, h)
    raise ValueError(f"Unknown resize_mode: {mode}")

def to_chw_array(img_l: Image.Image, spec: dict) -> np.ndarray:
    """Step 2: PIL 'L' → normalised float32 array (3, H, W)."""
    arr = np.asarray(img_l, dtype=np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=0)
    mean = np.asarray(spec["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(spec["std"], dtype=np.float32)[:, None, None]
    return (arr - mean) / std

def preprocess(img: Image.Image, spec: dict) -> np.ndarray:
    """Full pipeline for inference: returns (1, 3, H, W) float32."""
    return to_chw_array(resize_gray(img, spec), spec)[None]
