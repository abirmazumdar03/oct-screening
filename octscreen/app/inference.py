"""UI-independent screening core: model package + policy in
→ screening result dict out.

Used by the Gradio UI, CLI and tests.
"""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from octscreen.app.safety import check_image
from octscreen.app.triage import decide, validate_policy
from octscreen.common.model_meta import load_meta
from octscreen.common.preprocess import preprocess


def _softmax(z, temperature=1.0):
    z = np.asarray(z, dtype=np.float64) / float(temperature)
    e = np.exp(z - z.max())
    return e / e.sum()


class Screener:
    def __init__(
        self,
        package_dir,
        policy,
        enable_heatmap=True,
        threads=0,
        heatmap_alpha=0.45,
    ):
        self.dir = Path(package_dir)
        self.meta = load_meta(self.dir)
        self.classes = list(self.meta["class_names"])
        self.spec = self.meta["preprocess"]

        c = self.meta["classifier"]

        if c["format"] != "onnx":
            raise ValueError(
                f"Unsupported classifier format {c['format']!r} "
                "(this app runs ONNX)"
            )

        so = ort.SessionOptions()

        if int(threads) > 0:
            so.intra_op_num_threads = int(threads)

        self.sess = ort.InferenceSession(
            str(self.dir / c["file"]),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

        self.in_name = c["input_name"]
        self.out_name = c["output_name"]

        self.output_type = c["output_type"]
        self.temperature = float(c["temperature"])

        self.policy = policy
        validate_policy(policy, self.classes)

        self.placeholder = bool(self.meta.get("placeholder", False))
        self.alpha = float(heatmap_alpha)

        self.explainer = None

        if enable_heatmap:
            from octscreen.explain import build_explainer

            self.explainer = build_explainer(self.meta, self.dir)

        h, w = self.spec["image_size"]

        # Warm up ONNX Runtime.
        self.sess.run(
            [self.out_name],
            {
                self.in_name: np.zeros(
                    (1, 3, h, w),
                    np.float32,
                )
            },
        )

    def probabilities(self, img):
        x = preprocess(img, self.spec)

        t0 = time.perf_counter()

        out = self.sess.run(
            [self.out_name],
            {self.in_name: x},
        )[0][0]

        ms = (time.perf_counter() - t0) * 1e3

        if self.output_type == "logits":
            p = _softmax(out, self.temperature)
        else:
            p = np.asarray(out, dtype=np.float64)

        return p, ms

    def screen(self, img):
        t_start = time.perf_counter()

        res = {
            "status": "ok",
            "issues": [],
            "placeholder": self.placeholder,
            "package": (
                f"{self.meta['package_name']} "
                f"v{self.meta['package_version']}"
            ),
        }

        g = self.policy.guard

        if g.enabled:
            res["issues"] = check_image(img, g)

            if res["issues"] and str(g.mode) == "block":
                res["status"] = "rejected"
                return res

        p, model_ms = self.probabilities(img)

        k = int(p.argmax())
        conf = float(p[k])
        name = self.classes[k]

        res.update(
            class_idx=k,
            class_name=name,
            confidence=conf,
            probs={
                c: float(v)
                for c, v in zip(self.classes, p)
            },
            uncertain=(
                conf < float(self.policy.confidence_threshold)
            ),
            level=decide(
                name,
                conf,
                self.policy,
                forced_review=bool(res["issues"]),
            ),
            model_ms=round(model_ms, 1),
        )

        if self.explainer is not None:
            from octscreen.explain.overlay import overlay

            t0 = time.perf_counter()

            try:
                res["heatmap"] = overlay(
                    img,
                    self.explainer.explain(img, k),
                    self.alpha,
                )

                res["heatmap_ms"] = round(
                    (time.perf_counter() - t0) * 1e3,
                    1,
                )

            except Exception as e:
                # A heatmap problem must never hide the result.
                res["heatmap_error"] = str(e)

        res["total_ms"] = round(
            (time.perf_counter() - t_start) * 1e3,
            1,
        )

        return res