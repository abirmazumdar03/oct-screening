# octscreen/app/safety.py
"""Cheap input checks BEFORE the model runs. Heuristics, not guarantees: they catch obvious non-OCT inputs
(colour photos, tiny images, odd shapes) and, with `guard.structure`, grayscale images that lack the
look of an OCT B-scan (X-rays, black-and-white photos). Thresholds are fitted on real OCT scans, so a small,
measured share of real scans is also rejected (see docs/decisions.md)."""
import numpy as np
from PIL import Image

_BILINEAR = Image.BILINEAR


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

    st = g.get("structure", None)                      # optional block: absent or disabled = skipped
    if st is not None and bool(st.get("enabled", True)) and not issues:
        f = structure_scores(img, int(st.get("work_side", 512)))
        if (f["row_peak"] < float(st.min_row_peak) or f["median_brightness"] > float(st.max_median_brightness)
                or f["speckle"] < float(st.min_speckle)):
            issues.append(("not_oct_structure", {}))
    return issues


def _box_blur3(a):
    p = np.pad(a, 1, mode="edge")
    return sum(p[dy:dy + a.shape[0], dx:dx + a.shape[1]] for dy in range(3) for dx in range(3)) / 9.0


def structure_scores(img, work_side=512):
    """Three cheap measures of 'does this look like an OCT B-scan' (same definitions as the guard calibration):
    row_peak          : one bright horizontal band (the retina) on a dark background -> high on OCT
    median_brightness : OCT scans are mostly black                                   -> low on OCT
    speckle           : fine grainy texture relative to contrast                      -> X-rays / old photos are smooth
    """
    w, h = img.size
    s = work_side / max(w, h)
    g = np.asarray(img.convert("L").resize((max(16, round(w * s)), max(16, round(h * s))), _BILINEAR),
                   dtype=np.float32)
    sd = float(g.std()) + 1e-6
    rows = np.convolve(g.mean(axis=1), np.ones(5) / 5, mode="same")
    return {"row_peak": (float(rows.max()) - float(np.median(rows))) / sd,
            "median_brightness": float(np.percentile(g, 50)),
            "speckle": float((g - _box_blur3(g)).std()) / sd}
