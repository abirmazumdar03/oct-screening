# octscreen/optimize/export_onnx.py
"""Export training checkpoints (contract C3) to FP32 ONNX + sidecar, with a PyTorch↔ONNX parity check.

Usage:
  python -m octscreen.optimize.export_onnx --config configs/optimize.yaml                 # every model in `models:` that exists
  python -m octscreen.optimize.export_onnx --config configs/optimize.yaml --model teacher
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import torch

from octscreen.common.config import load_config
from octscreen.optimize.ort_utils import make_session, run, write_sidecar
from octscreen.training.models import load_checkpoint_model


def _export(model, dummy, out_path, opset, dynamic_batch, exporter="auto"):
    """Try the classic exporter first, then the newer 'dynamo' one. Returns which one worked."""
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=int(opset))
    if dynamic_batch:
        kw["dynamic_axes"] = {"input": {0: "batch"}, "logits": {0: "batch"}}
    order = {"auto": ["legacy", "dynamo"], "legacy": ["legacy"], "dynamo": ["dynamo"]}[exporter]
    errors = []
    for mode in order:
        try:
            if mode == "legacy":
                try:
                    torch.onnx.export(model, (dummy,), str(out_path), dynamo=False, **kw)
                except TypeError:                       # older torch without the `dynamo` argument
                    torch.onnx.export(model, (dummy,), str(out_path), **kw)
            else:
                torch.onnx.export(model, (dummy,), str(out_path), dynamo=True, external_data=False, **kw)
            return mode
        except Exception as e:                          # keep trying the next exporter
            errors.append(f"{mode}: {type(e).__name__}: {e}")
    raise RuntimeError("ONNX export failed:\n" + "\n".join(errors))


def parity_check(model, onnx_path, spec, n=4, seed=0):
    """Max |PyTorch - ONNX Runtime| over n random inputs (batch 1 each)."""
    h, w = spec["image_size"]
    rng = np.random.default_rng(seed)
    sess = make_session(onnx_path)
    worst = 0.0
    for _ in range(n):
        x = rng.standard_normal((1, 3, h, w)).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        worst = max(worst, float(np.abs(ref - run(sess, x)).max()))
    return worst


def export_one(name, ckpt_path, cfg):
    model, ck = load_checkpoint_model(ckpt_path, "cpu")
    spec = ck["preprocess"]
    h, w = spec["image_size"]
    out_dir = Path(cfg.export.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.fp32.onnx"
    mode = _export(model, torch.zeros(1, 3, h, w), out, cfg.export.opset, bool(cfg.export.dynamic_batch), str(cfg.export.exporter))
    onnx.checker.check_model(str(out))
    diff = parity_check(model, out, spec)
    vm = ck.get("val_metrics") or {}
    info = {
        "name": name, "precision": "fp32", "arch": ck["arch"], "num_classes": ck["num_classes"],
        "class_names": list(ck["class_names"]), "preprocess": spec,
        "source_checkpoint": str(ckpt_path), "checkpoint_epoch": ck.get("epoch"),
        "val_metrics_from_training": {k: vm.get(k) for k in ("accuracy", "macro_f1", "urgent_sensitivity")},
        "exporter": mode, "opset": int(cfg.export.opset), "parity_max_abs_diff": diff,
        "size_mb": round(out.stat().st_size / 1e6, 3),
    }
    write_sidecar(out, info)
    ok = diff <= float(cfg.export.parity_atol)
    print(f"[export] {name}: {out.name} {info['size_mb']:.2f} MB | exporter={mode} | parity={diff:.2e} {'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(f"Parity check failed for {name} ({diff:.2e} > {cfg.export.parity_atol})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optimize.yaml")
    ap.add_argument("--model", nargs="*", default=None, help="names from `models:` (default: all that exist)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    names = a.model or list(cfg.models.keys())
    for name in names:
        path = Path(str(cfg.models[name] or ""))
        if not path.is_file():
            print(f"[skip] {name}: checkpoint not found at {path}")
            continue
        export_one(name, path, cfg)


if __name__ == "__main__":
    main()