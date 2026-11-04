# tests/test_split_no_leak.py
from pathlib import Path

import pandas as pd
import pytest

CSV = Path("data/splits/splits_v1.csv")
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="splits not generated yet")


def _df():
    return pd.read_csv(CSV, dtype={"patient_id": str})


def test_no_patient_in_two_splits():
    assert (_df().groupby("patient_id")["split"].nunique() <= 1).all()


def test_unique_filenames_and_all_classes_everywhere():
    df = _df()
    assert df["filename"].is_unique
    assert set(df["split"]) == {"train", "val", "test"}
    for _, g in df.groupby("split"):
        assert g["label"].nunique() == df["label"].nunique()
