# octscreen/training/evaluate.py
"""Evaluate any checkpoint on any split → metrics_<split>.json + predictions_<split>.csv

Usage:  python -m octscreen.training.evaluate --checkpoint artifacts/runs/teacher_r50_v1/best.pt --split val
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from octscreen.common.config import load_config
from octscreen.common.metrics import compute_metrics
from octscreen.data.dataset import make_dataset
from octscreen.training.models import load_checkpoint_model
from octscreen.training.train import predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint_model(a.checkpoint, device)
    cfg.preprocess = ck["preprocess"]            # evaluate exactly as the model was trained
    cfg.data.class_names = ck["class_names"]
    classes = list(ck["class_names"])

    ds, df = make_dataset(cfg, a.split)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
    logits, y = predict(model, loader, device, use_amp=device.type == "cuda")
    probs = logits.softmax(1).numpy()
    pred = probs.argmax(1)
    m = compute_metrics(y.numpy(), pred, classes, list(cfg.urgent_classes))

    out = Path(a.out) if a.out else Path(a.checkpoint).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / f"metrics_{a.split}.json").write_text(json.dumps(m, indent=2))
    preds = df[["filename", "label", "patient_id"]].copy()
    preds["pred"] = [classes[i] for i in pred]
    for i, c in enumerate(classes):
        preds[f"p_{c}"] = probs[:, i]
    preds.to_csv(out / f"predictions_{a.split}.csv", index=False)
    print(json.dumps({k: m[k] for k in ("n", "accuracy", "macro_f1", "urgent_sensitivity") if k in m}, indent=2))


if __name__ == "__main__":
    main()
