# octscreen/data/discover.py
"""Scan a dataset folder (any layout) and build a table of images with label and patient id."""
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def iter_images(root, image_exts):
    exts = {e.lower() for e in image_exts}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            yield p


def discover(root, class_names, filename_regex, image_exts):
    """Returns (DataFrame[relpath, filename, label, patient_id, image_index], stats dict)."""
    root = Path(root)
    pattern = re.compile(filename_regex)
    classes = {c.upper(): c for c in class_names}
    rows, seen, stats = [], set(), Counter()
    for p in sorted(iter_images(root, image_exts)):
        m = pattern.match(p.name)
        if not m:
            stats["skipped_name_mismatch"] += 1
            continue
        label = m.group("label").upper()
        if label not in classes:
            stats["skipped_unknown_class"] += 1
            continue
        if p.name in seen:                          # duplicated folder trees → keep first copy
            stats["skipped_duplicate_filename"] += 1
            continue
        seen.add(p.name)
        if p.parent.name.strip().upper() != label:  # sanity check: folder vs file-name label
            stats["folder_label_mismatch"] += 1
        rows.append({
            "relpath": p.relative_to(root).as_posix(),
            "filename": p.name,
            "label": classes[label],
            "patient_id": m.group("patient"),
            "image_index": int(m.group("index")),
        })
    stats["kept"] = len(rows)
    return pd.DataFrame(rows), dict(stats)


def build_filename_index(root, image_exts):
    """filename → full path (used when data.path_resolution == 'filename')."""
    return {p.name: p for p in iter_images(root, image_exts)}
