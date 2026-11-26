# octscreen/app/ui_gradio.py
"""Bilingual Gradio front-end. All logic lives in inference/render/triage; this file only wires widgets.

Usage:  python -m octscreen.app [--set ui.default_language=hi] [--share]
"""
import argparse
import os
import sys
from itertools import chain, zip_longest
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")      # no telemetry; nothing phones home offline

import gradio as gr  # noqa: E402

from octscreen.app.i18n import Translator  # noqa: E402
from octscreen.app.inference import Screener  # noqa: E402
from octscreen.app.model_source import resolve_package  # noqa: E402
from octscreen.app.render import (about_markdown, banner_html, header_markdown, label_dict,  # noqa: E402
                                  result_markdown)
from octscreen.common.config import load_config  # noqa: E402

EXTS = (".jpeg", ".jpg", ".png")


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _examples(folder, n):
    """Up to n sample images, interleaved across class sub-folders (CNV, DME, DRUSEN, NORMAL, CNV, …)."""
    if not folder or not Path(str(folder)).is_dir():
        return []
    groups = {}
    for p in sorted(Path(str(folder)).rglob("*")):
        if p.suffix.lower() in EXTS:
            groups.setdefault(p.parent.name, []).append(str(p))
    mixed = [x for x in chain.from_iterable(zip_longest(*groups.values())) if x]
    return mixed[: int(n)]


def build_app(cfg):
    package = resolve_package(cfg.model)
    heatmap_on = _truthy(os.environ.get("OCT_ENABLE_HEATMAP", cfg.model.enable_heatmap))
    screener = Screener(package, cfg.policy, heatmap_on, int(cfg.model.threads), float(cfg.ui.heatmap_alpha))
    tr = Translator(cfg.ui.i18n_dir, list(cfg.ui.languages), str(cfg.ui.default_language))
    thr = float(cfg.policy.confidence_threshold)
    default_lang = str(cfg.ui.default_language)
    online = _truthy(os.environ.get("OCT_ONLINE_DEMO", "0"))

    def analyze(img, lang):
        if img is None:
            return tr.t(lang, "no_image"), None, None, None
        res = screener.screen(img)
        return result_markdown(res, lang, tr, thr), label_dict(res, lang, tr), res.get("heatmap"), res

    def auto_analyze(img, lang):
        if img is None:                       # image cleared → clear the result instead of nagging
            return "", None, None, None
        return analyze(img, lang)

    def texts(lang):
        """Every on-screen text in ONE language (the selected one)."""
        return {k: tr.t(lang, k) for k in ("upload_label", "analyze_button", "examples_label", "confidence",
                                           "heatmap_label", "about_heading")}

    t0 = texts(default_lang)
    with gr.Blocks(title=tr.both("title")) as demo:
        banner = gr.HTML(banner_html(default_lang, tr))
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                header = gr.Markdown(header_markdown(default_lang, tr, screener.placeholder, online))
            with gr.Column(scale=1, min_width=220):          # compact language switch, top right
                lang = gr.Radio(choices=[(tr.t(code, "language_name"), code) for code in cfg.ui.languages],
                                value=default_lang, label=tr.both("language_label"))   # bilingual on purpose
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label=t0["upload_label"], sources=["upload", "clipboard"])
                btn = gr.Button(t0["analyze_button"], variant="primary")
                examples = _examples(cfg.ui.examples_dir, cfg.ui.max_examples)
                ex = (gr.Examples(examples=examples, inputs=inp, label=t0["examples_label"], cache_examples=False)
                      if examples else None)
            with gr.Column(scale=1):
                out_md = gr.Markdown()
                out_label = gr.Label(num_top_classes=len(screener.classes), label=t0["confidence"])
                out_heat = gr.Image(label=t0["heatmap_label"], interactive=False,
                                    visible=screener.explainer is not None)
        with gr.Accordion(t0["about_heading"], open=False) as acc:
            about = gr.Markdown(about_markdown(screener.meta, default_lang, tr))
        state = gr.State(None)

        def relabel(res, lang):
            """Switch every text to the selected language. Re-labels only: the model is NOT re-run."""
            t = texts(lang)
            ups = [result_markdown(res, lang, tr, thr),
                   gr.update(value=label_dict(res, lang, tr), label=t["confidence"]),
                   about_markdown(screener.meta, lang, tr),
                   banner_html(lang, tr),
                   header_markdown(lang, tr, screener.placeholder, online),
                   gr.update(label=t["upload_label"]),
                   gr.update(value=t["analyze_button"]),
                   gr.update(label=t["heatmap_label"]),
                   gr.update(label=t["about_heading"])]
            if ex is not None:
                ups.append(gr.update(label=t["examples_label"]))
            return ups

        outputs = [out_md, out_label, out_heat, state]
        btn.click(analyze, [inp, lang], outputs)
        if bool(cfg.ui.auto_analyze):
            inp.change(auto_analyze, [inp, lang], outputs)        # fires on upload, paste AND sample-scan click
        relabel_out = [out_md, out_label, about, banner, header, inp, btn, out_heat, acc]
        if ex is not None:
            relabel_out.append(ex.dataset)
        lang.change(relabel, [state, lang], relabel_out)
    return demo


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):              # Windows consoles default to cp1252 → '→' / Hindi crash
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/app.yaml")
    ap.add_argument("--share", action="store_true", help="temporary public link (needs internet)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    demo = build_app(cfg)
    host = os.environ.get("OCT_HOST", str(cfg.ui.host))
    port = int(os.environ.get("PORT", cfg.ui.port))
    allowed = [str(Path(str(cfg.ui.examples_dir)).resolve())] if Path(str(cfg.ui.examples_dir)).is_dir() else []
    theme = gr.themes.Default(primary_hue=str(cfg.ui.get("theme_hue", "blue")))
    demo.queue().launch(server_name=host, server_port=port, share=a.share or bool(cfg.ui.share),
                        inbrowser=bool(cfg.ui.open_browser) and host in ("127.0.0.1", "localhost"),
                        allowed_paths=allowed, theme=theme)


if __name__ == "__main__":
    main()
