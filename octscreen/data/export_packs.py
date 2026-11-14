# octscreen/data/export_packs.py
"""Export small data packs for teammates.

Usage:  python -m octscreen.data.export_packs --pack sample --out /kaggle/working/packs
Packs:  sample = 10/class from TEST · calib = 128/class from TRAIN · test = ALL of TEST
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from octscreen.common.config import load_config
from octscreen.data.dataset import resolve_paths

PACKS = {"sample": ("test", 10, "sample_scans"), "calib": ("train", 128, "calib_pack"), "test": ("test", None, "test_pack") , "val": ("val", None, "val_pack")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--pack", choices=list(PACKS), required=True)
    ap.add_argument("--per-class", type=int, default=None, help="override the pack size")
    ap.add_argument("--out", default="artifacts/packs")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)

    split, n, name = PACKS[a.pack]
    n = a.per_class or n
    df = pd.read_csv(cfg.data.splits_csv, dtype={"patient_id": str})
    df = df[df["split"] == split].sample(frac=1.0, random_state=int(cfg.split.seed))
    if n:
        df = df.groupby("label", group_keys=False).head(n)
    df = df.reset_index(drop=True)

    dst_root = Path(a.out) / name
    for src, (_, r) in zip(resolve_paths(df, cfg), df.iterrows()):
        dst = dst_root / r["label"] / r["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    manifest = df.copy()
    manifest["relpath"] = manifest["label"] + "/" + manifest["filename"]
    manifest[["relpath", "filename", "label", "patient_id", "image_index", "split"]].to_csv(dst_root / "manifest.csv", index=False)
    if not a.no_zip:
        shutil.make_archive(str(dst_root), "zip", root_dir=dst_root)
    print(f"[pack] {name}: {len(df)} images → {dst_root}")


if __name__ == "__main__":
    main()
