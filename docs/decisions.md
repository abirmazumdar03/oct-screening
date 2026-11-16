# Decisions log
One line per decision: date, what, why.

- Wed 4 Nov: baseplate created (uv project, configs/data.yaml, octscreen/common). Patient-level split 80/10/10, seed 42, 224×224 squash.

MobileNetV3 failed 8-bit compression (27–41% agreement) because of its architecture. We switched the student to EfficientNet-Lite0, which was designed for 8-bit, and retrained it with the same teacher: 8-bit 3.80 MB, 99.5% agreement with the uncompressed model on 400 test scans.- 12 Nov: MobileNetV3-Small INT8 failed (27–41% agreement with FP32; teacher INT8 100%, so the pipeline was fine) → student switched to EfficientNet-Lite0.
- 12 Nov: Lite0 KD val macro-F1 0.919 vs baseline 0.888 (+3.1 pts). INT8 3.80 MB, 99.5% agreement with FP32 on abir-i3.
- 12 Nov: class proportions differ between splits (val: CNV 50.8%, DME 8.6%) because the split is by patient; kept as is.

## App · Poddar

- Thu 5 Nov: app core is UI-independent (`inference.py`, `render.py`); only `ui_gradio.py` imports Gradio.
- Thu 5 Nov: heatmaps use octscreen/app/heatmap.py (Grad-CAM on explainer.pt) until octscreen/explain is filled; then the app switches to it automatically.
- Thu 5 Nov: auto-analyse uses the image `change` event, so clicking a sample scan analyses it immediately (upload only would need an extra click).

## Guard (§7.1)

- Thu: guard defaults `min_side_px=128`, `aspect 0.5–4.0`, `max_color_diff=12`, mode `block`.
  Results: sample scans rejected **_/_** (target 0) · non-OCT images rejected **_/_** (`artifacts/guard_tests_results.csv`).
  Passed the guard (known limitation): _e.g. black-and-white photo of a desk_.

- Sat 14 Nov: **guard bug found in testing**: a colour selfie against grey walls scored 9.3 (< 12) and was screened as
  "CNV 100%, URGENT". Fix: also reject if > 2% of pixels are clearly coloured (chroma > 20), and lower max_color_diff to 4.
  Measured: selfie 24.5% coloured pixels → rejected; real OCT scans 0% (mean diff 0.0–0.7) → pass. Regression test added.
  Known limit stays: a black-and-white photo is grayscale and can still pass; the model is over-confident on such inputs.

## Confidence threshold (§9)

- Thu night (dev table): threshold _0.xx_ → _x %_ flagged, _y %_ silent urgent misses on val.
- Fri night (final table vX.Y.Z): threshold _0.xx_ → _x %_ flagged, _y %_ silent urgent misses on val.

## Offline end-to-end test (§11, Sat morning, Wi-Fi off)

| Laptop | Opens by itself | 4 classes OK | Hindi renders | Selfie / tiny refused | Displayed time | Abir's benchmark | Reopen OK |
|---|---|---|---|---|---|---|---|
| Poddar (i5, primary) | | | | | ms | ms | |
| Aditya (backup) | | | | | ms | ms | |
| Abir (i3, "frugal") | | | | | ms | ms | |

## Hosting

- Fri: host #1 HF Space `oct-screening-team/oct-screening`: _link, package version_
- Fri: host #2 Render: _link, package version_

- 16 Nov: **Input guard: added an "OCT structure" check** (`policy.guard.structure` in `configs/app.yaml`). Why: 9 grayscale non-OCT images (5 black-and-white photos, 4 X-rays) passed the colour check and were screened as OCT diseases. Three measures were compared on real OCT vs non-OCT (standalone EDA): bright-band score (one bright retinal band on a dark background), median brightness, and speckle. Limits were fitted on the VAL pack (7,980 real scans, each limit keeps 99.5%) and checked once on the TEST pack. Result: all 16 non-OCT test images caught (colour + structure); **real OCT wrongly rejected: 177/7,818 = 2.26% on test (109/7,980 = 1.37% on val)**. Rejected images are blocked with "doesn't look like an OCT scan; if it is, refer for manual review". Image size was deliberately NOT used as a rule: 496/512 px heights describe this dataset's scanner, not OCT in general.
