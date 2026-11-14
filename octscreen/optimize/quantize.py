# octscreen/optimize/quantize.py
"""FP32 ONNX → INT8 (static / dynamic) or FP16, as configured in configs/optimize.yaml → quant.

Usage:
  python -m octscreen.optimize.quantize --config configs/optimize.yaml --model student_kd
  python -m octscreen.optimize.quantize --config configs/optimize.yaml --model student_kd --tag int8_pct --set quant.calibrate_method=percentile
"""
import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
from omegaconf import OmegaConf
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
from PIL import Image

from octscreen.common.config import load_config, to_dict
from octscreen.common.preprocess import preprocess
from octscreen.data.dataset import load_split, resolve_paths
from octscreen.optimize.ort_utils import make_session, read_sidecar, run, write_sidecar

TAGS = {"int8_static": "int8", "int8_dynamic": "int8dyn", "fp16": "fp16"}
PRECISION = {"int8_static": "int8", "int8_dynamic": "int8", "fp16": "fp16"}
QT = {"quint8": QuantType.QUInt8, "qint8": QuantType.QInt8}
CM = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile, "entropy": CalibrationMethod.Entropy}


def _load(path, spec):
    with Image.open(path) as im:
        return preprocess(im, spec)


class ImageCalibrationReader(CalibrationDataReader):
    """Feeds preprocessed TRAIN images to the INT8 calibrator, a batch at a time."""

    def __init__(self, paths, spec, input_name, batch_size):
        self.paths, self.spec, self.input_name, self.bs = list(paths), spec, input_name, int(batch_size)
        self.rewind()

    def rewind(self):
        self._i = 0

    def get_next(self):
        if self._i >= len(self.paths):
            return None
        chunk = self.paths[self._i:self._i + self.bs]
        self._i += self.bs
        return {self.input_name: np.concatenate([_load(p, self.spec) for p in chunk], axis=0)}


def calib_paths(cfg):
    C = cfg.quant.calib
    if str(C.split) != "train":
        raise SystemExit("quant.calib.split must be 'train' (never calibrate on val/test)")
    c = OmegaConf.merge(cfg, {"data": {"root": C.root or cfg.data.root,
                                       "splits_csv": C.splits_csv or cfg.data.splits_csv,
                                       "limit_per_class": int(C.per_class)}})
    paths = resolve_paths(load_split(c, "train"), c)
    print(f"[calib] {len(paths)} train images from {c.data.root}")
    return paths


def _nodes_to_exclude(model, q):
    nodes = list(model.graph.node)
    names = []
    if q.exclude_first_conv:
        first = next((n for n in nodes if n.op_type == "Conv"), None)
        if first is not None:
            names.append(first.name)
    if q.exclude_last_linear:
        last = next((n for n in reversed(nodes) if n.op_type in ("Gemm", "MatMul", "Conv")), None)
        if last is not None:
            names.append(last.name)
    names += [str(n) for n in (q.get("nodes_to_exclude") or [])]
    if any(not n for n in names):
        print("[warn] some excluded nodes have empty names; use quant.nodes_to_exclude with explicit names")
    return [n for n in names if n]


def quantize_model(src, dst, q, paths, spec):
    mode = str(q.mode)
    if mode == "fp16":
        from onnxruntime.transformers.float16 import convert_float_to_float16
        onnx.save(convert_float_to_float16(onnx.load(str(src)), keep_io_types=True), str(dst))
    elif mode == "int8_dynamic":
        quantize_dynamic(str(src), str(dst), weight_type=QT[str(q.weight_type)],
                         per_channel=bool(q.per_channel), reduce_range=bool(q.reduce_range))
    elif mode == "int8_static":
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:   # Windows may keep file locks
            pre = Path(td) / "pre.onnx"
            try:
                from onnxruntime.quantization.shape_inference import quant_pre_process
                quant_pre_process(str(src), str(pre))
            except Exception as e:
                print(f"[warn] quant_pre_process skipped ({e}); quantizing the raw model")
                shutil.copy2(src, pre)
            model = onnx.load(str(pre))
            extra = {"CalibPercentile": float(q.percentile)} if str(q.calibrate_method) == "percentile" else {}
            quantize_static(
                str(pre), str(dst),
                ImageCalibrationReader(paths, spec, model.graph.input[0].name, q.calib.batch_size),
                quant_format=QuantFormat.QDQ if str(q.format) == "qdq" else QuantFormat.QOperator,
                activation_type=QT[str(q.activation_type)], weight_type=QT[str(q.weight_type)],
                per_channel=bool(q.per_channel), reduce_range=bool(q.reduce_range),
                calibrate_method=CM[str(q.calibrate_method)],
                nodes_to_exclude=_nodes_to_exclude(model, q), extra_options=extra)
    else:
        raise ValueError(f"Unknown quant.mode: {mode}")


def quick_agreement(src, dst, paths, spec, n=64):
    """% of images where the compressed model predicts the same class as FP32 (fast sanity check)."""
    paths = list(paths)[:n]
    if not paths:
        return None
    a, b = make_session(src), make_session(dst)
    same = [int(run(a, _load(p, spec)).argmax() == run(b, _load(p, spec)).argmax()) for p in paths]
    return round(100.0 * sum(same) / len(same), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optimize.yaml")
    ap.add_argument("--model", required=True, help="name from `models:`, e.g. student_kd")
    ap.add_argument("--tag", default=None, help="output tag (default from mode: int8 / int8dyn / fp16)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    q = cfg.quant
    if str(q.mode) == "fp32":
        print("[quant] mode=fp32: nothing to do (use <model>.fp32.onnx directly)")
        return
    src = Path(cfg.export.out_dir) / f"{a.model}.fp32.onnx"
    info = read_sidecar(src)
    dst = src.with_name(f"{a.model}.{a.tag or TAGS[str(q.mode)]}.onnx")
    paths = calib_paths(cfg)                     # also used for the quick agreement check
    t0 = time.time()
    quantize_model(src, dst, q, paths, info["preprocess"])
    agree = quick_agreement(src, dst, paths, info["preprocess"])
    new = dict(info, precision=PRECISION[str(q.mode)], quant=to_dict(q), source_fp32=src.name,
               size_mb=round(dst.stat().st_size / 1e6, 3), quant_agreement_pct_64_train_images=agree)
    write_sidecar(dst, new)
    print(f"[quant] {dst.name}: {info['size_mb']:.2f} MB → {new['size_mb']:.2f} MB "
          f"({info['size_mb'] / new['size_mb']:.1f}× smaller) | agreement with FP32 on 64 train images: {agree}% | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()