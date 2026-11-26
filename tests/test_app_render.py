# tests/test_app_render.py
import re
from pathlib import Path

import pytest

from octscreen.app.i18n import Translator
from octscreen.app.render import about_markdown, banner_html, header_markdown

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
META = {"package_name": "oct-student-kd", "package_version": "1.1.1", "classifier": {"precision": "int8"},
        "metrics": {"size_mb": 3.801, "test_accuracy": 0.952, "test_macro_f1": 0.928, "urgent_sensitivity": 0.969,
                    "latency_ms_p50": 13.97, "latency_device": "abir-i3 (Intel(R) Core(TM) i3-5005U CPU @ 2.00GHz)"}}


@pytest.fixture(scope="module")
def tr():
    return Translator(Path("configs/i18n"), ["en", "hi"], "en")


@pytest.mark.parametrize("lang", ["en", "hi"])
def test_about_panel_has_no_speed_or_machine_line(tr, lang):
    md = about_markdown(META, lang, tr)
    assert "13.97" not in md and "i3" not in md and " ms" not in md


def test_header_and_banner_follow_the_selected_language_only(tr):
    en = header_markdown("en", tr, online=True) + banner_html("en", tr)
    hi = header_markdown("hi", tr, online=True)
    assert not DEVANAGARI.search(en)                       # English view: no Hindi text
    assert DEVANAGARI.search(hi) and tr.t("en", "title") not in hi
    assert "ZEISS" in banner_html("hi", tr)


def test_ui_builds_and_language_switch_matches_its_outputs(monkeypatch):
    """Smoke test: the whole Gradio app builds, and the language switch returns one value per output."""
    gr = pytest.importorskip("gradio")
    if not Path("artifacts/model_package/model_meta.json").exists():
        pytest.skip("no model package unpacked")
    monkeypatch.setenv("OCT_ENABLE_HEATMAP", "0")
    from octscreen.app.ui_gradio import build_app
    from octscreen.common.config import load_config
    demo = build_app(load_config("configs/app.yaml"))
    assert isinstance(demo, gr.Blocks)
    fns = [f for f in getattr(demo, "fns", {}).values() if getattr(getattr(f, "fn", None), "__name__", "") == "relabel"]
    if not fns:
        pytest.skip("this Gradio version doesn't expose event functions")
    for lang in ("en", "hi"):
        assert len(fns[0].fn(None, lang)) == len(fns[0].outputs)
