# tools/make_placeholder.py
"""Placeholder checkpoints (random weights, contract C3) + a placeholder model package (C5).

Lets Abir and Poddar build and test everything before real models exist. NEVER report numbers from it.
Usage:  uv run python tools/make_placeholder.py            (≈ 1 minute on a laptop, no internet needed)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import timm
import torch

from octscreen.common.config import load_config, to_dict


def make_checkpoint(arch, cfg, path):
    classes = list(cfg.data.class_names)
    model = timm.create_model(arch, pretrained=False, num_classes=len(classes))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "state_dict": model.state_dict(), "arch": arch, "num_classes": len(classes),
                "class_names": classes, "preprocess": to_dict(cfg.preprocess), "epoch": -1,
                "val_metrics": {}, "config": {"placeholder": True}}, path)
    print(f"[placeholder] {path}")


def sh(*args):
    print(">", " ".join(args))
    subprocess.run([sys.executable, *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/placeholder")
    ap.add_argument("--teacher-arch", default="resnet50.tv2_in1k")
    ap.add_argument("--student-arch", default="mobilenetv3_small_100.lamb_in1k")
    ap.add_argument("--skip-teacher", action="store_true", help="faster: no 94 MB random teacher")
    a = ap.parse_args()
    cfg = load_config("configs/data.yaml")
    out = Path(a.out)
    ck = {"student_base": out / "runs/student_base/best.pt", "student_kd": out / "runs/student_kd/best.pt"}
    if not a.skip_teacher:
        ck["teacher"] = out / "runs/teacher/best.pt"
        make_checkpoint(a.teacher_arch, cfg, ck["teacher"])
    make_checkpoint(a.student_arch, cfg, ck["student_base"])
    make_checkpoint(a.student_arch, cfg, ck["student_kd"])

    common = [f"export.out_dir={(out / 'onnx').as_posix()}", f"benchmark.out_dir={(out / 'benchmark').as_posix()}"]
    common += [f"models.{k}={v.as_posix()}" for k, v in ck.items()]
    if a.skip_teacher:
        common.append("models.teacher=__none__")
    sh("-m", "octscreen.optimize.export_onnx", "--config", "configs/optimize.yaml", "--set", *common)
    sh("-m", "octscreen.optimize.package", "--config", "configs/optimize.yaml", "--set", *common,
       "package.classifier=student_kd.fp32", f"package.out_dir={(out / 'model_package').as_posix()}",
       "package.name=PLACEHOLDER-random-weights", "package.version=0.1.0", "package.placeholder=true")
    print(f"\nDone. Give Poddar: {out / 'PLACEHOLDER-random-weights-v0.1.0.zip'}")


if __name__ == "__main__":
    main()