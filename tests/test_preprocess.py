# tests/test_preprocess.py
import numpy as np
from PIL import Image

from octscreen.common.preprocess import preprocess

SPEC = {"image_size": [224, 224], "resize_mode": "squash", "color": "grayscale_to_rgb",
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def test_shape_dtype_and_determinism():
    img = Image.fromarray((np.random.default_rng(0).random((496, 768)) * 255).astype(np.uint8))
    a, b = preprocess(img, SPEC), preprocess(img, SPEC)
    assert a.shape == (1, 3, 224, 224) and a.dtype == np.float32
    assert np.array_equal(a, b)


def test_rgb_and_gray_inputs_give_same_tensor():
    gray = Image.fromarray((np.random.default_rng(1).random((300, 500)) * 255).astype(np.uint8))
    assert np.allclose(preprocess(gray, SPEC), preprocess(gray.convert("RGB"), SPEC), atol=1e-6)


def test_pad_mode_shape():
    img = Image.new("L", (1000, 300), 128)
    assert preprocess(img, dict(SPEC, resize_mode="pad")).shape == (1, 3, 224, 224)