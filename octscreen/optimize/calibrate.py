# octscreen/optimize/calibrate.py
"""Temperature scaling + confidence-threshold table, fitted on the VAL split only (never test).

Usage:  python -m octscreen.optimize.calibrate --config configs/optimize.yaml [--model student_kd.int8]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from octscreen.common.config import load_config
from octscreen.optimize.benchmark import eval_onnx
from octscreen.optimize.ort_utils import list_models, read_sidecar, softmax


def nll(logits, y, T):
    p = softmax(logits, T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def ece(probs, y, bins=15):
    """Expected calibration error: average gap between confidence and accuracy (0 = perfectly calibrated)."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(logits, y, t_min, t_max, steps):
    grid = np.geomspace(float(t_min), float(t_max), int(steps))
    losses = [nll(logits, y, t) for t in grid]
    return float(grid[int(np.argmin(losses))])


def threshold_table(probs, y, urgent_idx, thresholds):
    conf, pred = probs.max(1), probs.argmax(1)
    true_u, pred_u = np.isin(y, urgent_idx), np.isin(pred, urgent_idx)
    rows = []
    for t in thresholds:
        keep = conf >= float(t)
        rows.append({
            "threshold": float(t),
            "flagged_uncertain_pct": round(100 * float(1 - keep.mean()), 2),
            "accuracy_on_confident_pct": round(100 * float((pred[keep] == y[keep]).mean()), 2) if keep.any() else None,
            # truly urgent, NOT flagged as uncertain, and predicted as non-urgent → the dangerous case
            "silent_urgent_miss_pct": round(100 * float((true_u & keep & ~pred_u).sum() / max(int(true_u.sum()), 1)), 3),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optimize.yaml")
    ap.add_argument("--model", default=None, help="ONNX stem (default: package.classifier)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    C = cfg.calibration
    if str(C.split) != "val":
        raise SystemExit("calibration.split must be 'val'")
    stem = a.model or str(cfg.package.classifier)
    path = list_models(cfg.export.out_dir, [stem])[stem]
    classes = read_sidecar(path)["class_names"]
    out = Path(cfg.benchmark.out_dir)
    lp, yp = out / f"logits_val_{stem}.npy", out / "labels_val.npy"
    if lp.exists() and yp.exists():
        logits, y = np.load(lp), np.load(yp)
    else:
        logits, y, _, _ = eval_onnx(path, cfg, "val")

    T = fit_temperature(logits, y, C.t_min, C.t_max, C.t_steps) if C.enabled else 1.0
    p_before, p_after = softmax(logits, 1.0), softmax(logits, T)
    urgent = [classes.index(c) for c in cfg.urgent_classes if c in classes]
    table = threshold_table(p_after, y, urgent, list(C.thresholds))
    res = {"model": stem, "split": "val", "n": int(len(y)), "temperature": round(T, 4),
           "nll_before": round(nll(logits, y, 1.0), 4), "nll_after": round(nll(logits, y, T), 4),
           "ece_before": round(ece(p_before, y, C.ece_bins), 4), "ece_after": round(ece(p_after, y, C.ece_bins), 4),
           "threshold_table": table}
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = ["| Threshold | Flagged uncertain | Accuracy on the rest | Silent urgent misses |", "|---|---|---|---|"]
    md += [f"| {r['threshold']:.2f} | {r['flagged_uncertain_pct']}% | {r['accuracy_on_confident_pct']}% | {r['silent_urgent_miss_pct']}% |" for r in table]
    (out / "threshold_table.md").write_text(f"# Threshold table ({stem}, VAL, T={T:.3f})\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    print(f"[calib] T={T:.3f} | ECE {res['ece_before']:.4f} → {res['ece_after']:.4f} | NLL {res['nll_before']:.4f} → {res['nll_after']:.4f}")
    print("\n".join(md))


if __name__ == "__main__":
    main()