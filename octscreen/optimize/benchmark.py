# octscreen/optimize/benchmark.py
"""Benchmark: accuracy (Kaggle CPU), latency (each laptop), agreement, and the final report (contract C7).

Usage:
  python -m octscreen.optimize.benchmark accuracy --config configs/optimize.yaml --split val
  python -m octscreen.optimize.benchmark accuracy --config configs/optimize.yaml --split test      # FINAL, run ONCE
  python -m octscreen.optimize.benchmark latency  --config configs/optimize.yaml --images artifacts/packs/sample_scans --machine abir-i3
  python -m octscreen.optimize.benchmark agree    --config configs/optimize.yaml --a student_kd.fp32 --b student_kd.int8 --pack artifacts/packs/test_pack
  python -m octscreen.optimize.benchmark report   --config configs/optimize.yaml --split test
"""
import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from octscreen.common.config import load_config, to_dict
from octscreen.common.metrics import compute_metrics
from octscreen.common.preprocess import preprocess
from octscreen.data.dataset import load_split, resolve_paths
from octscreen.optimize.ort_utils import list_models, make_session, read_sidecar, run, supports_batching
from octscreen.optimize.sysinfo import system_info

IMAGE_EXTS = (".jpeg", ".jpg", ".png")


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(path, spec):
    with Image.open(path) as im:
        return preprocess(im, spec)[0]


# ── accuracy ─────────────────────────────────────────────────────────────────
def eval_onnx(path, cfg, split, batch_size=32, decode_threads=4, threads=0):
    info = read_sidecar(path)
    spec, classes = info["preprocess"], list(info["class_names"])
    c = OmegaConf.merge(cfg, {"data": {"class_names": classes}})
    df = load_split(c, split)
    paths = resolve_paths(df, c)
    y = df["label"].map({k: i for i, k in enumerate(classes)})
    if y.isna().any():
        raise ValueError(f"labels not in {classes}: {sorted(set(df.loc[y.isna(), 'label']))}")
    sess = make_session(path, threads)
    bs = int(batch_size) if supports_batching(sess) else 1
    chunks = []
    with ThreadPoolExecutor(max_workers=int(decode_threads)) as pool:
        for s in tqdm(range(0, len(paths), bs), desc=Path(path).stem, leave=False):
            x = np.stack(list(pool.map(lambda p: _load(p, spec), paths[s:s + bs])))
            chunks.append(run(sess, x))
    return np.concatenate(chunks).astype(np.float32), y.to_numpy().astype(np.int64), classes, df


def _print_quant_drops(results, max_drop):
    for stem, m in results.items():
        base, _, tag = stem.partition(".")
        ref = results.get(f"{base}.fp32")
        if tag != "fp32" and ref:
            drop = ref["macro_f1"] - m["macro_f1"]
            flag = "⚠ ABOVE LIMIT → see decision tree §6.4" if drop > max_drop else "ok"
            print(f"[quant-drop] {stem}: macro-F1 {ref['macro_f1']:.4f} → {m['macro_f1']:.4f} (drop {drop * 100:.2f} pts) {flag}")


def cmd_accuracy(cfg, a):
    out = Path(cfg.benchmark.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = list_models(cfg.export.out_dir, a.models)
    if a.split == "test":
        print("=" * 72 + "\n FINAL TEST-SET EVALUATION: run once, after every choice is frozen.\n" + "=" * 72)
        with open(out / "test_runs_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"utc": _utc(), "models": list(models)}) + "\n")
    B = cfg.benchmark
    results = {}
    for stem, path in models.items():
        t0 = time.time()
        logits, y, classes, df = eval_onnx(path, cfg, a.split, B.eval_batch, B.decode_threads, B.threads)
        m = compute_metrics(y, logits.argmax(1), classes, list(cfg.urgent_classes))
        info = read_sidecar(path)
        m.update(model=stem, name=info["name"], precision=info["precision"], arch=info["arch"],
                 size_mb=round(path.stat().st_size / 1e6, 3), split=a.split,
                 n_patients=int(df["patient_id"].nunique()), eval_seconds=round(time.time() - t0, 1), utc=_utc())
        results[stem] = m
        np.save(out / f"logits_{a.split}_{stem}.npy", logits)
        np.save(out / f"labels_{a.split}.npy", y)
        print(f"[acc:{a.split}] {stem:22s} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"urgent_sens={m.get('urgent_sensitivity', float('nan')):.4f} size={m['size_mb']:.2f}MB ({m['eval_seconds']}s)")
    _print_quant_drops(results, float(cfg.quant.max_f1_drop))
    path = out / f"accuracy_{a.split}.json"
    merged = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    merged.update(results)
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[ok] {path}")


# ── latency ──────────────────────────────────────────────────────────────────
def _stats(ms):
    a = np.asarray(ms)
    return {k: round(float(v), 3) for k, v in
            {"p50": np.percentile(a, 50), "p90": np.percentile(a, 90), "mean": a.mean(), "std": a.std(), "min": a.min()}.items()}


def _load_images(folder, limit):
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)[:int(limit)]
    if not files:
        raise SystemExit(f"No images found in {folder}")
    imgs = []
    for p in files:
        with Image.open(p) as im:
            im.load()
            imgs.append(im.copy())
    return imgs


def cmd_latency_one(cfg, a):
    """Runs in a FRESH process per model, so memory numbers aren't polluted by other models."""
    import psutil
    proc = psutil.Process()
    baseline = proc.memory_info().rss
    L = cfg.benchmark.latency
    images = _load_images(a.images, L.max_images)
    spec = read_sidecar(a.onnx)["preprocess"]
    sess = make_session(a.onnx, L.threads)
    for i in range(int(L.warmup)):
        run(sess, preprocess(images[i % len(images)], spec))
    model_ms, e2e_ms = [], []
    for i in range(int(L.runs)):
        im = images[i % len(images)]
        t0 = time.perf_counter()
        x = preprocess(im, spec)
        t1 = time.perf_counter()
        run(sess, x)
        t2 = time.perf_counter()
        model_ms.append((t2 - t1) * 1e3)
        e2e_ms.append((t2 - t0) * 1e3)
    mi = proc.memory_info()
    print(json.dumps({"model_ms": _stats(model_ms), "end_to_end_ms": _stats(e2e_ms),
                      "rss_mb": round(mi.rss / 1e6, 1), "peak_mb": round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1),
                      "baseline_mb": round(baseline / 1e6, 1), "n_images": len(images)}))


def cmd_latency(cfg, a):
    info = system_info()
    machine = re.sub(r"[^A-Za-z0-9_.-]", "_", a.machine or info["machine"])
    if info["on_battery"]:
        print("[warn] laptop is ON BATTERY: plug in the charger for fair numbers (result is still saved, flagged)")
    out = Path(cfg.benchmark.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {"machine": machine, "utc": _utc(), "system": info, "settings": to_dict(cfg.benchmark.latency),
           "what": "batch 1; model_ms = ONNX Runtime run only; end_to_end_ms = preprocessing + run (image already decoded)",
           "models": {}}
    for stem, path in list_models(cfg.export.out_dir, a.models).items():
        cmd = [sys.executable, "-m", "octscreen.optimize.benchmark", "latency-one", "--config", a.config,
               "--onnx", str(path), "--images", a.images, "--set", *a.set]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"[error] {stem}:\n{p.stderr[-2000:]}")
            continue
        r = json.loads(p.stdout.strip().splitlines()[-1])
        r["size_mb"] = round(path.stat().st_size / 1e6, 3)
        res["models"][stem] = r
        print(f"[lat:{machine}] {stem:22s} p50={r['model_ms']['p50']:.1f} ms (e2e {r['end_to_end_ms']['p50']:.1f} ms) peak RAM {r['peak_mb']:.0f} MB")
    path = out / f"latency_{machine}.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[ok] {path}  ← send this file to Abir")


# ── agreement (FP32 vs compressed, on this machine) ──────────────────────────
def cmd_agree(cfg, a):
    pack = Path(a.pack)
    manifest = pack / "manifest.csv"
    split = str(pd.read_csv(manifest, nrows=1)["split"].iloc[0])
    c = OmegaConf.merge(cfg, {"data": {"root": str(pack), "splits_csv": str(manifest), "limit_per_class": a.limit}})
    models = list_models(cfg.export.out_dir, [a.a, a.b])
    la, _, _, _ = eval_onnx(models[a.a], c, split, 16, 2, 0)
    lb, _, _, _ = eval_onnx(models[a.b], c, split, 16, 2, 0)
    pct = 100.0 * float((la.argmax(1) == lb.argmax(1)).mean())
    machine = re.sub(r"[^A-Za-z0-9_.-]", "_", system_info()["machine"])
    res = {"a": a.a, "b": a.b, "pack": str(pack), "n": int(len(la)), "agreement_pct": round(pct, 2), "machine": machine, "utc": _utc()}
    out = Path(cfg.benchmark.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"agree_{a.b}_{machine}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[agree] {a.a} vs {a.b} on {len(la)} images ({machine}): {pct:.2f}% same prediction")


# ── report (contract C7) ─────────────────────────────────────────────────────
def _confusion_png(m, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_title(f"{m['model']} ({m['split']})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cmd_report(cfg, a):
    out = Path(cfg.benchmark.out_dir)
    acc_file = out / f"accuracy_{a.split}.json"
    if not acc_file.exists():
        raise SystemExit(f"{acc_file} missing: run `benchmark accuracy --split {a.split}` first")
    acc = json.loads(acc_file.read_text(encoding="utf-8"))
    lats = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(out.glob("latency_*.json"))]
    order = [s for s in cfg.benchmark.report_order if s in acc] + sorted(s for s in acc if s not in cfg.benchmark.report_order)
    ref = acc.get(str(cfg.benchmark.reference_model), {}).get("size_mb")

    head = ["Model", "Precision", "Size (MB)", "× smaller", "Accuracy", "Macro-F1", "Urgent sensitivity"]
    head += [f"p50 ms · {l['machine']}" for l in lats]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in order:
        m = acc[s]
        row = [s, m["precision"], f"{m['size_mb']:.2f}", f"{ref / m['size_mb']:.1f}×" if ref else "–",
               f"{m['accuracy'] * 100:.2f}%", f"{m['macro_f1'] * 100:.2f}%", f"{m.get('urgent_sensitivity', float('nan')) * 100:.2f}%"]
        row += [f"{l['models'][s]['model_ms']['p50']:.1f}" if s in l["models"] else "–" for l in lats]
        lines.append("| " + " | ".join(row) + " |")

    any_m = acc[order[0]]
    title = "FINAL (test split, evaluated once)" if a.split == "test" else "DEVELOPMENT numbers (val split), NOT final"
    md = [f"# Benchmark: {title}", "",
          f"Split: **{a.split}**, {any_m['n']} images from {any_m['n_patients']} patients (patient-level split; no patient overlaps training).", "",
          *lines, "",
          "## Machines", "", "| Machine | CPU | Cores (phys/log) | RAM | On battery | ONNX Runtime |", "|---|---|---|---|---|---|"]
    for l in lats:
        s = l["system"]
        md.append(f"| {l['machine']} | {s['cpu']} | {s['physical_cores']}/{s['logical_cores']} | {s['ram_gb']} GB | "
                  f"{'YES (flagged)' if s['on_battery'] else 'no'} | {s['onnxruntime']} |")
    md += ["", "## How these numbers were measured", "",
           "- Accuracy metrics: ONNX Runtime (CPU) on every model, same images, same preprocessing code as the app.",
           "- Size: the `.onnx` file on disk (MB = 10^6 bytes). `× smaller` = reference size ÷ model size, both measured.",
           f"- Latency: batch 1, {cfg.benchmark.latency.warmup} warm-up runs, median (p50) of {cfg.benchmark.latency.runs} runs, "
           "model inference only (end-to-end incl. preprocessing is in benchmark.json), charger plugged in unless flagged.",
           "- Urgent sensitivity: % of truly urgent scans (CNV, DME) predicted as an urgent class.",
           "- Dataset: Kermany et al., Cell 2018 (public research data, mostly one OCT scanner brand)."]
    (out / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "benchmark.json").write_text(json.dumps({"split": a.split, "accuracy": acc, "latency": lats}, indent=2), encoding="utf-8")
    for s in order:
        _confusion_png(acc[s], list(acc[s]["per_class"].keys()), out / f"confusion_{a.split}_{s}.png")
    print("\n".join(lines))
    print(f"[ok] {out / 'benchmark.md'}")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/optimize.yaml")
    common.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("accuracy", parents=[common])
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--models", nargs="*", default=None)
    p = sub.add_parser("latency", parents=[common])
    p.add_argument("--images", required=True)
    p.add_argument("--machine", default=None)
    p.add_argument("--models", nargs="*", default=None)
    p = sub.add_parser("latency-one", parents=[common])
    p.add_argument("--onnx", required=True)
    p.add_argument("--images", required=True)
    p = sub.add_parser("agree", parents=[common])
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--pack", required=True)
    p.add_argument("--limit", type=int, default=None, help="max images per class")
    p = sub.add_parser("report", parents=[common])
    p.add_argument("--split", choices=["val", "test"], default="test")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    {"accuracy": cmd_accuracy, "latency": cmd_latency, "latency-one": cmd_latency_one,
     "agree": cmd_agree, "report": cmd_report}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()