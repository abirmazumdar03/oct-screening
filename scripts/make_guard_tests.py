# scripts/make_guard_tests.py
"""Synthesize obvious NON-OCT images for the guard test set (§7.1). This SUPPLEMENTS real phone photos
(desk, face, document, screenshot), which you should add yourself to artifacts/guard_tests/.

Usage:  python scripts/make_guard_tests.py [--out artifacts/guard_tests] [--from-scan path/to/one_oct.jpeg]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/guard_tests")
    ap.add_argument("--from-scan", default=None, help="a real OCT scan → also writes a tiny 64×64 crop of it")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    yy, xx = np.mgrid[0:480, 0:640]
    grad = np.stack([xx / 640 * 255, yy / 480 * 255, np.full_like(xx, 140)], -1).astype(np.uint8)
    Image.fromarray(grad).save(out / "colour_gradient.png")                       # → not_grayscale

    Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)).save(out / "colour_noise.png")

    skin = np.zeros((600, 480, 3), np.uint8)
    skin[...] = (224, 172, 140)                                                   # selfie-like skin tone
    im = Image.fromarray(skin)
    ImageDraw.Draw(im).ellipse((120, 150, 360, 450), fill=(198, 134, 100))
    im.save(out / "skin_tone_selfie_like.jpg")

    diagram = Image.new("RGB", (800, 500), "white")                               # colourful diagram
    d = ImageDraw.Draw(diagram)
    for i, c in enumerate(["red", "green", "blue", "orange"]):
        d.rectangle((40 + i * 190, 150, 190 + i * 190, 350), fill=c)
    diagram.save(out / "colour_diagram.png")

    Image.new("L", (64, 64), 90).save(out / "tiny_64x64.png")                    # → too_small
    Image.new("L", (300, 2400), 200).save(out / "very_tall_screenshot.png")       # → bad_aspect
    Image.new("L", (3000, 300), 120).save(out / "very_wide_strip.png")            # → bad_aspect

    if a.from_scan:
        with Image.open(a.from_scan) as scan:
            scan.convert("L").crop((0, 0, 64, 64)).save(out / "tiny_crop_of_real_scan.png")
    (out / "not_an_image.jpg").write_bytes(b"this is not a JPEG")                 # → unreadable
    print(f"[ok] synthetic guard tests → {out}  (now add ~5 real phone photos / screenshots too)")


if __name__ == "__main__":
    main()
