# octscreen/app/safety.py
"""Cheap input checks BEFORE the model runs. Heuristics, not guarantees: they catch obvious non-OCT inputs
(colour photos, tiny images, odd shapes). Grayscale non-OCT images can still pass; the confidence
threshold is the second line of defence."""
import numpy as np


def check_image(img, g):
    """Return a list of (issue_code, params) tuples; empty list = OK."""
    issues = []
    w, h = img.size
    if min(w, h) < int(g.min_side_px):
        issues.append(("too_small", {"w": w, "h": h}))
    ratio = w / max(h, 1)
    if not float(g.aspect_min) <= ratio <= float(g.aspect_max):
        issues.append(("bad_aspect", {"ratio": round(ratio, 2)}))
    rgb = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.int16)      # small copy → fast
    color = float((np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()) / 2)
    if color > float(g.max_color_diff):
        issues.append(("not_grayscale", {"score": round(color, 1)}))
    return issues
