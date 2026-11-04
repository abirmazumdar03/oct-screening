# tests/test_baseplate.py: proves the shared baseplate works on every machine
import numpy as np
from PIL import Image

from octscreen.common.config import load_config
from octscreen.common.metrics import compute_metrics
from octscreen.common.preprocess import preprocess


def test_data_config_loads_and_overrides():
    cfg = load_config("configs/data.yaml", ["split.seed=7"])
    assert list(cfg.data.class_names) == ["CNV", "DME", "DRUSEN", "NORMAL"]
    assert cfg.split.seed == 7


def test_preprocess_shape():
    cfg = load_config("configs/data.yaml")
    img = Image.new("L", (768, 496), 128)
    x = preprocess(img, dict(cfg.preprocess))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32


def test_metrics_urgent_sensitivity():
    m = compute_metrics([0, 1, 2, 3], [0, 1, 3, 3], ["CNV", "DME", "DRUSEN", "NORMAL"], ["CNV", "DME"])
    assert m["urgent_sensitivity"] == 1.0 and m["n"] == 4
