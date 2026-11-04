# tests/test_split_logic.py
import numpy as np
import pandas as pd

from octscreen.data.split import check_no_leak, patient_split


def test_patient_split_synthetic():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(600):
        label = rng.choice(["CNV", "DME", "DRUSEN", "NORMAL"])
        for i in range(int(rng.integers(1, 8))):
            rows.append({"filename": f"{label}-{p}-{i}.jpeg", "label": label, "patient_id": str(p)})
    df = patient_split(pd.DataFrame(rows), val_frac=0.1, test_frac=0.1, seed=42)
    check_no_leak(df)
    frac = df["split"].value_counts(normalize=True)
    assert 0.05 < frac["test"] < 0.15 and 0.05 < frac["val"] < 0.15
    assert set(df["split"]) == {"train", "val", "test"}
