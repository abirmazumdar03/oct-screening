# Decisions log
One line per decision: date, what, why.

- Wed 4 Nov: baseplate created (uv project, configs/data.yaml, octscreen/common). Patient-level split 80/10/10, seed 42, 224×224 squash.

MobileNetV3 failed 8-bit compression (27–41% agreement) because of its architecture. We switched the student to EfficientNet-Lite0, which was designed for 8-bit, and retrained it with the same teacher: 8-bit 3.80 MB, 99.5% agreement with the uncompressed model on 400 test scans.