
"""Screen every image in a folder from the terminal → table + CSV. Works offline, no browser needed.

Usage:  python -m octscreen.app.cli --images artifacts/packs/sample_scans [--lang hi] [--heatmap] [--out results.csv]
"""
import argparse
import csv
from pathlib import Path

from PIL import Image

from octscreen.app.i18n import Translator
from octscreen.app.inference import Screener
from octscreen.app.model_source import resolve_package
from octscreen.common.config import load_config

EXTS = (".jpeg", ".jpg", ".png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/app.yaml")
    ap.add_argument("--images", required=True)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--heatmap", action="store_true")
    ap.add_argument("--out", default="artifacts/app_cli_results.csv")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    lang = a.lang or str(cfg.ui.default_language)
    tr = Translator(cfg.ui.i18n_dir, list(cfg.ui.languages), str(cfg.ui.default_language))
    screener = Screener(resolve_package(cfg.model), cfg.policy, enable_heatmap=a.heatmap, threads=int(cfg.model.threads))

    rows = []
    for p in sorted(q for q in Path(a.images).rglob("*") if q.suffix.lower() in EXTS):
        try:
            with Image.open(p) as im:
                im.load()
                res = screener.screen(im)
        except Exception:
            res = {"status": "rejected", "issues": [("unreadable", {})]}
        rows.append({
            "file": p.as_posix(), "folder": p.parent.name, "status": res["status"],
            "prediction": res.get("class_name", ""), "confidence": round(res.get("confidence", 0.0), 4),
            "triage": tr.t(lang, "triage." + res["level"]) if "level" in res else "",
            "issues": "; ".join(code for code, _ in res["issues"]), "model_ms": res.get("model_ms", ""),
        })
        print(f"{rows[-1]['status']:8s} {rows[-1]['prediction']:7s} {rows[-1]['confidence']:.2f}  {rows[-1]['triage']}  ← {p.name}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig: Hindi opens correctly in Excel
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"[ok] {len(rows)} images → {a.out}")


if __name__ == "__main__":
    main()
