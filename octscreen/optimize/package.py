# octscreen/optimize/package.py
"""Build the model package (contract C5) that Poddar's app consumes, then zip it with a SHA-256.

Usage:  python -m octscreen.optimize.package --config configs/optimize.yaml
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from octscreen.common.config import load_config
from octscreen.common.model_meta import SCHEMA_VERSION, validate_meta
from octscreen.optimize.ort_utils import read_sidecar


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _strip_checkpoint(src, dst):
    """Keep only what the explainer needs (no optimizer, no config) → small file."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    keep = {k: ck[k] for k in ("format_version", "state_dict", "arch", "num_classes", "class_names", "preprocess") if k in ck}
    torch.save(keep, dst)
    return ck["arch"]


def _pick_latency(bench_dir, stem, machine=None):
    """Latency p50 for this model: the requested machine, else the SLOWEST measured one (conservative)."""
    cands = []
    for f in sorted(Path(bench_dir).glob("latency_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        m = d.get("models", {}).get(stem)
        if m:
            cands.append((d["machine"], d["system"].get("cpu"), m["model_ms"]["p50"]))
    if machine:
        cands = [c for c in cands if c[0] == machine] or cands
    if not cands:
        return None, None
    mach, cpu, p50 = max(cands, key=lambda c: c[2])
    return round(p50, 2), f"{mach} ({cpu}), batch 1, model only"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optimize.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P, bench = cfg.package, Path(cfg.benchmark.out_dir)

    src = Path(cfg.export.out_dir) / f"{P.classifier}.onnx"
    info = read_sidecar(src)
    out = Path(P.out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(src, out / "classifier.onnx")

    temperature = 1.0
    calib = bench / "calibration.json"
    if calib.exists():
        c = json.loads(calib.read_text(encoding="utf-8"))
        if c.get("model") == P.classifier:
            temperature = float(c["temperature"])
        else:
            print(f"[warn] calibration.json is for {c.get('model')}, not {P.classifier}: using temperature 1.0")

    metrics = {"test_accuracy": None, "test_macro_f1": None, "urgent_sensitivity": None,
               "size_mb": round((out / "classifier.onnx").stat().st_size / 1e6, 3),
               "latency_ms_p50": None, "latency_device": None}
    acc_file = bench / "accuracy_test.json"
    if acc_file.exists():
        m = json.loads(acc_file.read_text(encoding="utf-8")).get(P.classifier)
        if m:
            metrics.update(test_accuracy=round(m["accuracy"], 4), test_macro_f1=round(m["macro_f1"], 4),
                           urgent_sensitivity=round(m.get("urgent_sensitivity", float("nan")), 4))
    metrics["latency_ms_p50"], metrics["latency_device"] = _pick_latency(bench, P.classifier, P.get("latency_machine"))

    explainer = None
    if P.include_explainer:
        ckpt = cfg.models.get(str(P.explainer_from))
        if ckpt and Path(str(ckpt)).is_file():
            arch = _strip_checkpoint(str(ckpt), out / "explainer.pt")
            if arch != info["arch"]:
                print(f"[warn] explainer arch {arch} != classifier arch {info['arch']}: heatmaps may not match predictions")
            explainer = {"type": "gradcam_torch", "file": "explainer.pt", "arch": arch, "target_layer": str(P.target_layer)}
        else:
            print(f"[warn] explainer checkpoint not found ({ckpt}): package will have no heatmaps")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "package_name": str(P.name),
        "package_version": str(P.version),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "placeholder": bool(P.placeholder),
        "class_names": list(info["class_names"]),
        "preprocess": info["preprocess"],
        "classifier": {"format": "onnx", "file": "classifier.onnx", "input_name": "input", "output_name": "logits",
                       "output_type": "logits", "precision": info["precision"], "temperature": temperature},
        "explainer": explainer,
        "metrics": metrics,
        "source": {"checkpoint": info.get("source_checkpoint"), "onnx": src.name, "git_commit": _git_commit()},
    }
    errors = validate_meta(meta, out)
    if errors:
        raise SystemExit("Package invalid:\n - " + "\n - ".join(errors))
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "README.txt").write_text(
        f"{meta['package_name']} v{meta['package_version']} ({meta['created_utc']})\n"
        f"{'PLACEHOLDER: random weights, NOT a real model.' if meta['placeholder'] else 'OCT screening classifier (4 classes).'}\n"
        f"Classifier: {info['arch']} {info['precision']}, {metrics['size_mb']} MB. Temperature: {temperature:.3f}.\n"
        f"Metrics (null = not measured yet): {json.dumps(metrics)}\n"
        "Screening aid for research/demo only, not a diagnosis. Data: Kermany et al., Cell 2018.\n",
        encoding="utf-8")

    archive = Path(shutil.make_archive(str(out.parent / f"{P.name}-v{P.version}"), "zip", root_dir=out))
    (archive.parent / (archive.name + ".sha256")).write_text(f"{_sha256(archive)}  {archive.name}\n", encoding="utf-8")
    print(f"[package] {out} → {archive.name} | {metrics}")


if __name__ == "__main__":
    main()