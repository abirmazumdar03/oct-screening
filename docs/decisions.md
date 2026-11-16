@'
# Decisions Log

One line per decision: date, what, why.

- Wed 4 Nov: Baseplate created (`uv` project, `configs/data.yaml`, `octscreen/common`). Patient-level split of 80/10/10 with random seed 42 and input resolution of 224 × 224.

- 12 Nov: MobileNetV3-Small failed INT8 quantisation, achieving only 27–41% agreement with its FP32 version. The teacher INT8 model achieved 100% agreement, indicating that the quantisation pipeline was functioning correctly. The student architecture was therefore changed to EfficientNet-Lite0 and retrained using the same ResNet-50 teacher.

- 12 Nov: EfficientNet-Lite0 knowledge-distilled student achieved a validation macro-F1 of 0.919 compared with 0.888 for the baseline, an improvement of 3.1 percentage points. The INT8 model size was 3.80 MB and its agreement with the uncompressed model was 99.5% on 400 test scans.

- 12 Nov: Class proportions differed between the patient-level splits (validation: CNV 50.8%, DME 8.6%). The patient-level split was retained to prevent patient overlap between training, validation and testing.
'@ | Set-Content -Path docs/decisions.md -Encoding utf8
## Input Safety Guard Test — 15 November 2026

**Test set:** 60 non-OCT images across six categories.

| Category | Total | Flagged | Passed |
|---|---:|---:|---:|
| Colourful diagrams | 10 | 9 | 1 |
| Computer screenshots | 10 | 4 | 6 |
| Desk photographs | 10 | 7 | 3 |
| Document photographs | 10 | 5 | 5 |
| Face photographs | 10 | 5 | 5 |
| Grayscale images | 10 | 0 | 10 |
| **Total** | **60** | **30** | **30** |

- Overall rejection rate: 50%.
- All 30 flagged images were flagged by the grayscale heuristic.
- No images were flagged for minimum dimensions or aspect ratio.
- All 10 grayscale non-OCT images passed, demonstrating a limitation of the current heuristic.
- The test used non-OCT images only. Genuine OCT acceptance has not yet been evaluated.
- Guard thresholds remain unchanged pending validation against genuine OCT samples.

