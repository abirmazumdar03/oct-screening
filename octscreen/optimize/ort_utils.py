# octscreen/optimize/ort_utils.py
"""ONNX Runtime helpers + sidecar metadata. Imports only numpy + onnxruntime."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def make_session(path, threads=0):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads and int(threads) > 0:
        so.intra_op_num_threads = int(threads)
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def io_names(sess):
    return sess.get_inputs()[0].name, sess.get_outputs()[0].name


def supports_batching(sess):
    """True if the input's batch dimension is dynamic (a name or None instead of a fixed int)."""
    return not isinstance(sess.get_inputs()[0].shape[0], int)


def run(sess, x):
    i, o = io_names(sess)
    return sess.run([o], {i: np.ascontiguousarray(x, dtype=np.float32)})[0]


def softmax(z, temperature=1.0):
    z = np.asarray(z, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sidecar_path(onnx_path):
    return Path(onnx_path).with_suffix(".json")          # student_kd.int8.onnx → student_kd.int8.json


def write_sidecar(onnx_path, info):
    sidecar_path(onnx_path).write_text(json.dumps(info, indent=2), encoding="utf-8")


def read_sidecar(onnx_path):
    p = sidecar_path(onnx_path)
    if not p.exists():
        raise FileNotFoundError(f"Sidecar {p} missing: every .onnx must come from export_onnx/quantize")
    return json.loads(p.read_text(encoding="utf-8"))


def list_models(onnx_dir, stems=None):
    """{stem: path} for every .onnx with a sidecar in onnx_dir (optionally only the requested stems)."""
    d = Path(onnx_dir)
    found = {p.stem: p for p in sorted(d.glob("*.onnx")) if sidecar_path(p).exists()}
    if stems:
        missing = [s for s in stems if s not in found]
        if missing:
            raise SystemExit(f"Not found in {d}: {missing}. Available: {sorted(found)}")
        return {s: found[s] for s in stems}
    return found