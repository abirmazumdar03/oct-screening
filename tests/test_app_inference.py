# tests/test_app_inference.py
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("onnxruntime")
PKG = Path("artifacts/model_package")


@pytest.mark.skipif(not (PKG / "model_meta.json").exists(), reason="no model package unpacked")
def test_screen_returns_a_complete_result():
    from octscreen.app.inference import Screener
    from octscreen.common.config import load_config
    cfg = load_config("configs/app.yaml")
    s = Screener(PKG, cfg.policy, enable_heatmap=False)
    img = Image.fromarray((np.random.default_rng(0).random((496, 768)) * 255).astype(np.uint8))
    res = s.screen(img)
    assert res["status"] == "ok"
    assert abs(sum(res["probs"].values()) - 1.0) < 1e-5
    assert res["level"] in ("urgent", "routine", "none", "review")
    assert set(res["probs"]) == set(s.classes)


@pytest.mark.skipif(not (PKG / "model_meta.json").exists(), reason="no model package unpacked")
def test_colour_photo_is_rejected_end_to_end_and_renders_in_hindi():
    from octscreen.app.i18n import Translator
    from octscreen.app.inference import Screener
    from octscreen.app.render import label_dict, result_markdown
    from octscreen.common.config import load_config
    cfg = load_config("configs/app.yaml")
    s = Screener(PKG, cfg.policy, enable_heatmap=False)
    arr = np.zeros((480, 640, 3), np.uint8)
    arr[..., 0] = 230
    res = s.screen(Image.fromarray(arr))
    assert res["status"] == "rejected"
    tr = Translator("configs/i18n", ["en", "hi"])
    md = result_markdown(res, "hi", tr, 0.7)
    assert "इस छवि की जाँच नहीं की जा सकती" in md and label_dict(res, "hi", tr) is None


def test_overlay_returns_rgb_image_of_the_same_size():
    from octscreen.app.heatmap import overlay
    img = Image.new("L", (300, 200), 128)
    out = overlay(img, np.random.default_rng(0).random((7, 7)), 0.45)
    assert out.size == (300, 200) and out.mode == "RGB"


def test_model_source_skips_unconfigured_and_reports_errors(tmp_path):
    from omegaconf import OmegaConf

    from octscreen.app.model_source import resolve_package
    cfg = OmegaConf.create({"cache_dir": str(tmp_path), "sources": [
        {"type": "local", "path": str(tmp_path / "missing")},
        {"type": "url_zip", "url": None, "sha256": None},
        {"type": "hf_repo", "repo_id": None, "revision": "main"}]})
    with pytest.raises(RuntimeError, match="local"):
        resolve_package(cfg)
