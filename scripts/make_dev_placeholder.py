# scripts/make_dev_placeholder.py
"""Write a PLACEHOLDER model package (contract C5) with RANDOM weights, for wiring tests only.

Prefer Abir's real placeholder (tools/make_placeholder.py on his branch: random timm weights + explainer.pt).
Use this one only if his zip hasn't reached you yet: no heatmap, but it passes his validator (common/model_meta.py). Needs numpy only: the ONNX file is
encoded by hand (tiny graph: GlobalAveragePool + GlobalMaxPool → Concat → Flatten → Gemm → 4 logits).

Usage:  python scripts/make_dev_placeholder.py [--out artifacts/model_package] [--seed 0]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from octscreen.common.model_meta import validate_meta          # Abir's contract C5 validator

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]


# ---- minimal protobuf writer (just what an ONNX ModelProto needs) ---------------------------------------
def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _key(field, wire):
    return _varint((field << 3) | wire)


def _int(field, v):
    return _key(field, 0) + _varint(int(v))


def _bytes(field, b):
    b = b.encode("utf-8") if isinstance(b, str) else b
    return _key(field, 2) + _varint(len(b)) + b


def _tensor(name, arr):              # TensorProto: dims=1, data_type=2 (FLOAT=1), name=8, raw_data=9
    arr = np.ascontiguousarray(arr, dtype="<f4")
    return b"".join(_int(1, d) for d in arr.shape) + _int(2, 1) + _bytes(8, name) + _bytes(9, arr.tobytes())


def _value_info(name, dims):         # ValueInfoProto{name, TypeProto{tensor_type{elem_type, shape{dim...}}}}
    dim_msgs = b"".join(_bytes(1, _bytes(2, d) if isinstance(d, str) else _int(1, d)) for d in dims)
    tensor_type = _int(1, 1) + _bytes(2, dim_msgs)
    return _bytes(1, name) + _bytes(2, _bytes(1, tensor_type))


def _node(op, inputs, outputs, name, attrs=()):
    msg = b"".join(_bytes(1, i) for i in inputs) + b"".join(_bytes(2, o) for o in outputs)
    msg += _bytes(3, name) + _bytes(4, op)
    for aname, aval in attrs:        # AttributeProto INT: name=1, i=3, type=20 (INT=2)
        msg += _bytes(5, _bytes(1, aname) + _int(3, aval) + _int(20, 2))
    return msg


def build_onnx(weights, bias, h, w):
    nodes = [
        _node("GlobalAveragePool", ["input"], ["gap"], "gap"),
        _node("GlobalMaxPool", ["input"], ["gmp"], "gmp"),
        _node("Concat", ["gap", "gmp"], ["cat"], "cat", [("axis", 1)]),
        _node("Flatten", ["cat"], ["flat"], "flat", [("axis", 1)]),
        _node("Gemm", ["flat", "W", "B"], ["logits"], "fc"),
    ]
    graph = b"".join(_bytes(1, n) for n in nodes) + _bytes(2, "placeholder")
    graph += _bytes(5, _tensor("W", weights)) + _bytes(5, _tensor("B", bias))
    graph += _bytes(11, _value_info("input", ["batch", 3, h, w]))
    graph += _bytes(12, _value_info("logits", ["batch", len(bias)]))
    opset = _bytes(1, "") + _int(2, 13)
    return _int(1, 7) + _bytes(2, "octscreen-dev-placeholder") + _bytes(7, graph) + _bytes(8, opset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/model_package")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--size", type=int, nargs=2, default=[224, 224], metavar=("H", "W"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    w = rng.normal(0, 0.6, (6, len(CLASSES))).astype(np.float32)
    b = rng.normal(0, 0.3, len(CLASSES)).astype(np.float32)
    onnx_bytes = build_onnx(w, b, *a.size)
    meta = {
        "schema_version": 1,
        "package_name": "PLACEHOLDER-random-weights", "package_version": "0.1.0-dev", "placeholder": True,
        "class_names": CLASSES,
        "preprocess": {"image_size": list(a.size), "resize_mode": "squash", "color": "grayscale_to_rgb",
                       "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "classifier": {"format": "onnx", "file": "classifier.onnx", "input_name": "input", "output_name": "logits",
                       "output_type": "logits", "precision": "fp32", "temperature": 1.0},
        "explainer": None,
        "metrics": {"size_mb": round(len(onnx_bytes) / 1e6, 4), "test_accuracy": None, "test_macro_f1": None,
                    "urgent_sensitivity": None, "latency_ms_p50": None, "latency_device": None},
    }
    (out / "classifier.onnx").write_bytes(onnx_bytes)
    problems = validate_meta(meta, out)
    if problems:
        raise SystemExit("placeholder invalid:\n - " + "\n - ".join(problems))
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] placeholder package → {out}  ({len(onnx_bytes)} bytes, random weights, testing only)")


if __name__ == "__main__":
    main()
