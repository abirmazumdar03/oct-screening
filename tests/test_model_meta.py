# tests/test_model_meta.py
import copy
from pathlib import Path

import pytest

from octscreen.common.model_meta import load_meta, validate_meta

GOOD = {
    "schema_version": 1, "package_name": "t", "package_version": "0.0.1", "class_names": ["CNV", "DME", "DRUSEN", "NORMAL"],
    "preprocess": {"image_size": [224, 224], "resize_mode": "squash", "color": "grayscale_to_rgb",
                   "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "classifier": {"format": "onnx", "file": "classifier.onnx", "input_name": "input", "output_name": "logits",
                   "output_type": "logits", "precision": "int8", "temperature": 1.0},
    "explainer": None,
}


def test_good_meta_is_valid():
    assert validate_meta(GOOD) == []


@pytest.mark.parametrize("key", ["classifier", "class_names", "preprocess"])
def test_missing_top_key_is_reported(key):
    bad = copy.deepcopy(GOOD)
    bad.pop(key)
    assert validate_meta(bad)


def test_bad_temperature_is_reported():
    bad = copy.deepcopy(GOOD)
    bad["classifier"]["temperature"] = 0
    assert validate_meta(bad)


def test_placeholder_package_loads_if_built():
    pkg = Path("artifacts/placeholder/model_package")
    if not (pkg / "model_meta.json").exists():
        pytest.skip("placeholder package not built (run tools/make_placeholder.py)")
    meta = load_meta(pkg)
    assert meta["placeholder"] is True