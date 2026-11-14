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
from octscreen.app.render import about_markdown, label_dict, result_markdown  # noqa: E402
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

    def relabel(res, lang):
        return result_markdown(res, lang, tr, thr), label_dict(res, lang, tr), about_markdown(screener.meta, lang, tr)

    with gr.Blocks(title=tr.both("title")) as demo:
        header = f"# {tr.both('title')}\n{tr.both('subtitle', sep='  ·  ')}"
        if screener.placeholder:
            header += f"\n\n> {tr.both('placeholder_banner', sep='  /  ')}"
        if online:
            header += f"\n\n> {tr.both('online_notice', sep='  /  ')}"
        gr.Markdown(header)
        lang = gr.Radio(choices=[(tr.t(code, "language_name"), code) for code in cfg.ui.languages],
                        value=default_lang, label=tr.both("language_label"))
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label=tr.both("upload_label"), sources=["upload", "clipboard"])
                btn = gr.Button(tr.both("analyze_button"), variant="primary")
                examples = _examples(cfg.ui.examples_dir, cfg.ui.max_examples)
                if examples:
                    gr.Examples(examples=examples, inputs=inp, label=tr.both("examples_label"), cache_examples=False)
            with gr.Column(scale=1):
                out_md = gr.Markdown()
                out_label = gr.Label(num_top_classes=len(screener.classes), label=tr.both("confidence"))
                out_heat = gr.Image(label=tr.both("heatmap_label"), interactive=False,
                                    visible=screener.explainer is not None)
        with gr.Accordion(tr.both("about_heading"), open=False):
            about = gr.Markdown(about_markdown(screener.meta, default_lang, tr))
        state = gr.State(None)

        outputs = [out_md, out_label, out_heat, state]
        btn.click(analyze, [inp, lang], outputs)
        if bool(cfg.ui.auto_analyze):
            inp.change(auto_analyze, [inp, lang], outputs)        # fires on upload, paste AND sample-scan click
        lang.change(relabel, [state, lang], [out_md, out_label, about])   # re-label only, no model re-run
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
    demo.queue().launch(server_name=host, server_port=port, share=a.share or bool(cfg.ui.share),
                        inbrowser=bool(cfg.ui.open_browser) and host in ("127.0.0.1", "localhost"),
                        allowed_paths=allowed)


if __name__ == "__main__":
    main()
