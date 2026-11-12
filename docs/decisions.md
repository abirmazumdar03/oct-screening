# Decisions log
One line per decision: date, what, why.

- Wed 4 Nov: baseplate created (uv project, configs/data.yaml, octscreen/common). Patient-level split 80/10/10, seed 42, 224×224 squash.

MobileNetV3 failed 8-bit compression (27–41% agreement) because of its architecture. We switched the student to EfficientNet-Lite0, which was designed for 8-bit, and retrained it with the same teacher: 8-bit 3.80 MB, 99.5% agreement with the uncompressed model on 400 test scans.- 12 Nov: MobileNetV3-Small INT8 failed (27–41% agreement with FP32; teacher INT8 100%, so the pipeline was fine) → student switched to EfficientNet-Lite0.
- 12 Nov: Lite0 KD val macro-F1 0.919 vs baseline 0.888 (+3.1 pts). INT8 3.80 MB, 99.5% agreement with FP32 on abir-i3.
- 12 Nov: class proportions differ between splits (val: CNV 50.8%, DME 8.6%) because the split is by patient; kept as is.
