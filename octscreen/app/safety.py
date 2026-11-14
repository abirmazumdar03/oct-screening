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
    # 1) average channel difference: catches strongly coloured images
    mean_diff = float((np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()) / 2)
    # 2) share of clearly coloured pixels: catches MUTED photos (grey walls + a face) whose average stays low.
    #    Real OCT scans (grayscale) have ~0% such pixels; an indoor selfie had ~25%.
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    colour_frac = float((chroma > int(g.get("color_pixel_chroma", 20))).mean())
    if mean_diff > float(g.max_color_diff) or colour_frac > float(g.get("max_color_fraction", 0.02)):
        issues.append(("not_grayscale", {"score": round(max(mean_diff, 100 * colour_frac), 1)}))
    return issues
