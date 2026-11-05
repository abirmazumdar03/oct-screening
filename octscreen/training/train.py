# octscreen/training/train.py
"""Train a teacher or a student (optionally with knowledge distillation).

Usage:
  python -m octscreen.training.train --config configs/teacher.yaml
  python -m octscreen.training.train --config configs/student_kd.yaml --set kd.temperature=2 run.name=kd_T2
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from octscreen.common.config import load_config, to_dict
from octscreen.common.metrics import compute_metrics
from octscreen.common.seed import seed_everything
from octscreen.data.augment import build_train_augment
from octscreen.data.dataset import make_dataset
from octscreen.training.losses import class_weights, kd_loss
from octscreen.training.models import build_model, load_checkpoint_model, set_backbone_trainable, split_param_groups


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap.parse_args()


def make_scheduler(opt, steps_per_epoch, epochs, warmup_epochs, min_lr_ratio):
    """Linear warm-up, then cosine decay to min_lr_ratio × base LR (applied per step, per param group)."""
    total = max(1, steps_per_epoch * epochs)
    warm = int(steps_per_epoch * warmup_epochs)

    def factor(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        progress = min(1.0, (step - warm) / max(1, total - warm))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(opt, factor)


@torch.no_grad()
def predict(model, loader, device, use_amp=False, channels_last=False):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        all_logits.append(logits.float().cpu())
        all_y.append(y)
    return torch.cat(all_logits), torch.cat(all_y)


def save_checkpoint(path, model, cfg, classes, epoch, val_metrics, extra=None):
    obj = {
        "format_version": 1,
        "state_dict": model.state_dict(),
        "arch": cfg.model.arch,
        "num_classes": len(classes),
        "class_names": classes,
        "preprocess": to_dict(cfg.preprocess),
        "epoch": epoch,
        "val_metrics": val_metrics,
        "config": to_dict(cfg),
    }
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(int(cfg.run.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    channels_last = bool(cfg.train.get("channels_last", False)) and device.type == "cuda"
    classes = list(cfg.data.class_names)

    out = Path(cfg.run.out_dir) / cfg.run.name
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config_resolved.yaml")
    print(f"[run] {cfg.run.name} | arch={cfg.model.arch} | device={device} amp={use_amp} | out={out}", flush=True)

    # ── data ────────────────────────────────────────────────────────────
    train_ds, _ = make_dataset(cfg, "train", augment=build_train_augment(cfg.aug))
    val_ds, _ = make_dataset(cfg, "val", augment=None)
    nw = int(cfg.train.num_workers)
    loader_kw = dict(num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    bs = int(cfg.train.batch_size)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=len(train_ds) > bs, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, **loader_kw)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} batches/epoch={len(train_loader)}", flush=True)

    # ── model (+ optional teacher) ──────────────────────────────────────
    model = build_model(cfg.model.arch, len(classes), bool(cfg.model.pretrained), float(cfg.model.get("drop_rate", 0.0))).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    teacher = None
    if cfg.get("kd") is not None and cfg.kd.enabled:
        teacher, tck = load_checkpoint_model(cfg.kd.teacher_checkpoint, device)
        if list(tck["class_names"]) != classes:
            raise ValueError(f"Teacher classes {tck['class_names']} != {classes}")
        if tck["preprocess"] != to_dict(cfg.preprocess):
            print("[warn] teacher preprocessing differs from this config!", flush=True)
        if channels_last:
            teacher = teacher.to(memory_format=torch.channels_last)
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[kd] teacher={tck['arch']} (val macro-F1 {tck['val_metrics'].get('macro_f1', float('nan')):.4f}) "
              f"T={cfg.kd.temperature} alpha={cfg.kd.alpha}", flush=True)

    w = class_weights(train_ds.labels, len(classes), cfg.train.class_weighting)
    ce_fn = nn.CrossEntropyLoss(weight=None if w is None else w.to(device), label_smoothing=float(cfg.train.label_smoothing))
    opt = torch.optim.AdamW(split_param_groups(model, cfg.train.lr_backbone, cfg.train.lr_head, cfg.train.weight_decay))
    sched = make_scheduler(opt, len(train_loader), int(cfg.train.epochs), float(cfg.train.warmup_epochs), float(cfg.train.min_lr_ratio))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── resume ──────────────────────────────────────────────────────────
    start_epoch, best_score, bad_epochs = 0, -float("inf"), 0
    if cfg.train.get("resume"):
        ck = torch.load(cfg.train.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_score, bad_epochs = ck["epoch"] + 1, ck["best_score"], ck.get("bad_epochs", 0)
        print(f"[resume] from epoch {ck['epoch']} (best={best_score:.4f})", flush=True)

    monitor = str(cfg.train.monitor)
    freeze_n = int(cfg.train.get("freeze_backbone_epochs", 0))
    hist_path = out / "history.csv"

    # ── epochs ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, int(cfg.train.epochs)):
        set_backbone_trainable(model, epoch >= freeze_n)
        model.train()
        t0, n = time.time(), 0
        sums = {"loss": 0.0, "ce": 0.0, "kl": 0.0}
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                t_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
            if teacher is not None:
                loss, ce, kl = kd_loss(logits.float(), t_logits.float(), y, ce_fn, cfg.kd.temperature, float(cfg.kd.alpha))
            else:
                loss = ce_fn(logits.float(), y)
                ce, kl = loss.detach(), torch.zeros(())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
            scaler.step(opt)
            scaler.update()
            sched.step()
            b = y.size(0)
            n += b
            sums["loss"] += loss.item() * b
            sums["ce"] += ce.item() * b
            sums["kl"] += float(kl) * b
            if step % int(cfg.train.log_every) == 0:
                print(f"  ep{epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}", flush=True)

        logits_v, y_v = predict(model, val_loader, device, use_amp, channels_last)
        m = compute_metrics(y_v.numpy(), logits_v.argmax(1).numpy(), classes, list(cfg.urgent_classes))
        m["loss"] = F.cross_entropy(logits_v, y_v).item()
        row = {
            "epoch": epoch,
            "train_loss": sums["loss"] / max(n, 1), "train_ce": sums["ce"] / max(n, 1), "train_kl": sums["kl"] / max(n, 1),
            "val_loss": m["loss"], "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"],
            "val_urgent_sens": m.get("urgent_sensitivity", float("nan")),
            "lr_backbone": opt.param_groups[0]["lr"], "lr_head": opt.param_groups[1]["lr"],
            "epoch_sec": round(time.time() - t0, 1),
        }
        new_file = not hist_path.exists()
        with open(hist_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print("[epoch] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

        score = -m["loss"] if monitor == "loss" else m[monitor]
        if score > best_score:
            best_score, bad_epochs = score, 0
            save_checkpoint(out / "best.pt", model, cfg, classes, epoch, m)
            (out / "metrics_val.json").write_text(json.dumps(m, indent=2))
            print(f"  ↳ new best {monitor}={m.get(monitor, m['loss']):.4f} saved", flush=True)
        else:
            bad_epochs += 1
        save_checkpoint(out / "last.pt", model, cfg, classes, epoch, m, extra={
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
            "best_score": best_score, "bad_epochs": bad_epochs})
        if bad_epochs >= int(cfg.train.early_stopping_patience):
            print(f"[early-stop] no improvement for {bad_epochs} epochs", flush=True)
            break

    print(f"[done] best {monitor}={best_score:.4f} → {out / 'best.pt'}", flush=True)
    try:
        from octscreen.training.plots import plot_history
        plot_history(hist_path, out / "curves.png")
    except Exception as e:  # plotting must never kill a finished run
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
