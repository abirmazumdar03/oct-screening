# octscreen/explain/gallery.py
"""Heatmap gallery: N scans per class with true label, prediction, confidence and Grad-CAM overlay.

Usage:  python -m octscreen.explain.gallery --package artifacts/model_package --images artifacts/packs/sample_scans --per-class 3
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from octscreen.common.model_meta import load_meta
from octscreen.common.preprocess import preprocess
from octscreen.explain import build_explainer
from octscreen.explain.overlay import overlay
from octscreen.optimize.ort_utils import make_session, run, softmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="artifacts/model_package")
    ap.add_argument("--images", default="artifacts/packs/sample_scans")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--out", default="artifacts/benchmark/heatmap_gallery.png")
    a = ap.parse_args()

    pkg = Path(a.package)
    meta = load_meta(pkg)
    classes, spec, T = meta["class_names"], meta["preprocess"], meta["classifier"]["temperature"]
    sess = make_session(pkg / meta["classifier"]["file"])
    explainer = build_explainer(meta, pkg)
    if explainer is None:
        raise SystemExit("This package has no explainer")

    picks = []
    for c in classes:
        files = sorted(p for p in (Path(a.images) / c).glob("*") if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
        picks += [(c, p) for p in files[:a.per_class]]
    cols = a.per_class
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (true, path) in zip(axes.ravel(), picks):
        with Image.open(path) as im:
            im.load()
            probs = softmax(run(sess, preprocess(im, spec)), T)[0]
            k = int(probs.argmax())
            ax.imshow(overlay(im, explainer.explain(im, k), a.alpha))
        mark = "✓" if classes[k] == true else "✗"
        ax.set_title(f"{mark} true {true} · pred {classes[k]} ({probs[k]:.0%})", fontsize=9)
    fig.suptitle(f"Grad-CAM, {meta['package_name']} v{meta['package_version']}" + ("  [PLACEHOLDER]" if meta.get("placeholder") else ""))
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"[gallery] {a.out}")


if __name__ == "__main__":
    main()