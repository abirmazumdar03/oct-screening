
"""Bilingual Gradio front-end. All logic lives in inference/render/triage; this file only wires widgets.

Usage:  python -m octscreen.app [--set ui.default_language=hi] [--share]
"""
import argparse
import os
from itertools import chain, zip_longest
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from octscreen.app.i18n import Translator
from octscreen.app.inference import Screener
from octscreen.app.model_source import resolve_package
from octscreen.app.render import about_markdown, label_dict, result_markdown
from octscreen.common.config import load_config

EXTS = (".jpeg", ".jpg", ".png")


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _examples(folder, n):
    """Up to n sample images, interleaved across class sub-folders."""
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
    screener = Screener(
        package, cfg.policy, heatmap_on,
        int(cfg.model.threads), float(cfg.ui.heatmap_alpha)
    )
    tr = Translator(
        cfg.ui.i18n_dir, list(cfg.ui.languages),
        str(cfg.ui.default_language)
    )
    thr = float(cfg.policy.confidence_threshold)
    default_lang = str(cfg.ui.default_language)
    online = _truthy(os.environ.get("OCT_ONLINE_DEMO", "0"))

    def make_header(selected_lang):
        header = (
            f"# {tr.t(selected_lang, 'title')}\n"
            f"{tr.t(selected_lang, 'subtitle')}"
        )
        if screener.placeholder:
            header += f"\n\n> {tr.t(selected_lang, 'placeholder_banner')}"
        if online:
            header += f"\n\n> {tr.t(selected_lang, 'online_notice')}"
        return header

    def analyze(img, selected_lang):
        if img is None:
            return tr.t(selected_lang, "no_image"), None, None, None

        res = screener.screen(img)
        return (
            result_markdown(res, selected_lang, tr, thr),
            label_dict(res, selected_lang, tr),
            res.get("heatmap"),
            res,
        )

    def relabel(res, selected_lang):
        updates = [
            gr.update(value=make_header(selected_lang)),
            gr.update(label=tr.t(selected_lang, "language_label")),
            gr.update(label=tr.t(selected_lang, "upload_label")),
            gr.update(value=tr.t(selected_lang, "analyze_button")),
        ]

        if examples_widget is not None:
            updates.append(
                gr.update(label=tr.t(selected_lang, "examples_label"))
            )

        updates.extend([
            gr.update(label=tr.t(selected_lang, "confidence")),
            gr.update(label=tr.t(selected_lang, "heatmap_label")),
            gr.update(label=tr.t(selected_lang, "about_heading")),
            result_markdown(res, selected_lang, tr, thr) if res else "",
            label_dict(res, selected_lang, tr) if res else None,
            about_markdown(screener.meta, selected_lang, tr),
        ])
        return updates

    with gr.Blocks(title=tr.t(default_lang, "title")) as demo:
        header_md = gr.Markdown(make_header(default_lang))

        lang = gr.Radio(
            choices=[
                (tr.t(language, "language_name"), language)
                for language in cfg.ui.languages
            ],
            value=default_lang,
            label=tr.t(default_lang, "language_label"),
        )

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(
                    type="pil",
                    label=tr.t(default_lang, "upload_label"),
                    sources=["upload", "clipboard"],
                )
                btn = gr.Button(
                    tr.t(default_lang, "analyze_button"),
                    variant="primary",
                )

                examples = _examples(
                    cfg.ui.examples_dir, cfg.ui.max_examples
                )
                examples_widget = None

                if examples:
                    examples_widget = gr.Examples(
                        examples=examples,
                        inputs=inp,
                        label=tr.t(default_lang, "examples_label"),
                        cache_examples=False,
                    )

            with gr.Column(scale=1):
                out_md = gr.Markdown()
                out_label = gr.Label(
                    num_top_classes=len(screener.classes),
                    label=tr.t(default_lang, "confidence"),
                )
                out_heat = gr.Image(
                    label=tr.t(default_lang, "heatmap_label"),
                    interactive=False,
                    visible=screener.explainer is not None,
                )

        with gr.Accordion(
            tr.t(default_lang, "about_heading"), open=False
        ) as about_accordion:
            about = gr.Markdown(
                about_markdown(screener.meta, default_lang, tr)
            )

        state = gr.State(None)
        outputs = [out_md, out_label, out_heat, state]

        btn.click(analyze, [inp, lang], outputs)

        if bool(cfg.ui.auto_analyze):
            inp.upload(analyze, [inp, lang], outputs)

        language_outputs = [
            header_md, lang, inp, btn
        ]

        if examples_widget is not None:
            language_outputs.append(examples_widget.dataset)

        language_outputs.extend([
            out_label,
            out_heat,
            about_accordion,
            out_md,
            out_label,
            about,
        ])

        lang.change(
            relabel,
            inputs=[state, lang],
            outputs=language_outputs,
        )

    return demo

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/app.yaml")
    ap.add_argument(
        "--share", action="store_true",
        help="temporary public link (needs internet)"
    )
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.set)
    demo = build_app(cfg)

    host = os.environ.get("OCT_HOST", str(cfg.ui.host))
    port = int(os.environ.get("PORT", cfg.ui.port))

    demo.queue().launch(
        server_name=host,
        server_port=port,
        share=a.share or bool(cfg.ui.share),
        inbrowser=bool(cfg.ui.open_browser)
        and host in ("127.0.0.1", "localhost"),
    )


if __name__ == "__main__":
    main()
