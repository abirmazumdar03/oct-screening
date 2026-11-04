# octscreen/data/split.py
"""Create the patient-level train/val/test split.

Usage:  python -m octscreen.data.split --config configs/data.yaml
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from octscreen.common.config import load_config
from octscreen.data.discover import discover

COLUMNS = ["relpath", "filename", "label", "patient_id", "image_index", "split"]


def _take_one_fold(df, n_splits, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held = next(sgkf.split(df, df["label"], groups=df["patient_id"]))
    return df.index[held]


def patient_split(df, val_frac, test_frac, seed):
    df = df.reset_index(drop=True).copy()
    df["split"] = "train"
    test_idx = _take_one_fold(df, max(2, round(1 / test_frac)), seed)
    df.loc[test_idx, "split"] = "test"
    rest = df[df["split"] == "train"]
    val_idx = _take_one_fold(rest, max(2, round((1 - test_frac) / val_frac)), seed + 1)
    df.loc[val_idx, "split"] = "val"
    return df


def check_no_leak(df):
    per_patient = df.groupby("patient_id")["split"].nunique()
    leaked = per_patient[per_patient > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} patients appear in more than one split, e.g. {list(leaked.index[:5])}")


def check_all_classes_present(df, class_names):
    missing = [(s, c) for s in ("train", "val", "test") for c in class_names
               if not ((df["split"] == s) & (df["label"] == c)).any()]
    if missing:
        raise AssertionError(f"Empty split/class combinations: {missing}")


def summarize(df, cfg, stats):
    counts = df.pivot_table(index="split", columns="label", values="filename", aggfunc="count", fill_value=0)
    return {
        "version": cfg.split.version,
        "seed": int(cfg.split.seed),
        "val_frac": float(cfg.split.val_frac),
        "test_frac": float(cfg.split.test_frac),
        "n_images": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "images_per_split_class": {s: {c: int(v) for c, v in row.items()} for s, row in counts.iterrows()},
        "patients_per_split": {s: int(g["patient_id"].nunique()) for s, g in df.groupby("split")},
        "fingerprint_sha1": hashlib.sha1("\n".join(sorted(df["filename"])).encode()).hexdigest(),
        "discovery": stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)

    df, stats = discover(cfg.data.root, list(cfg.data.class_names), cfg.data.filename_regex, list(cfg.data.image_exts))
    print("[discover]", stats)
    if df.empty:
        raise SystemExit(f"No images found under {cfg.data.root} (check OCT_DATA_ROOT)")

    df = patient_split(df, float(cfg.split.val_frac), float(cfg.split.test_frac), int(cfg.split.seed))
    check_no_leak(df)
    check_all_classes_present(df, list(cfg.data.class_names))

    out_csv = Path(cfg.data.splits_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].sort_values(["split", "label", "filename"]).to_csv(out_csv, index=False)
    summary = summarize(df, cfg, stats)
    (out_csv.parent / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n_images", "n_patients", "patients_per_split")}, indent=2))
    print(json.dumps(summary["images_per_split_class"], indent=2))
    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
