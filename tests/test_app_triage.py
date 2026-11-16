
import pytest
from omegaconf import OmegaConf

from octscreen.app.triage import decide, validate_policy

POLICY = OmegaConf.create({
    "confidence_threshold": 0.7,
    "triage": {
        "CNV": "urgent",
        "DME": "urgent",
        "DRUSEN": "routine",
        "NORMAL": "none",
    },
})

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def test_levels():
    validate_policy(POLICY, CLASSES)

    assert decide("CNV", 0.95, POLICY) == "urgent"
    assert decide("NORMAL", 0.95, POLICY) == "none"
    assert decide("CNV", 0.50, POLICY) == "review"
    assert decide("NORMAL", 0.95, POLICY, forced_review=True) == "review"


def test_missing_class_fails_fast():
    with pytest.raises(ValueError):
        validate_policy(POLICY, CLASSES + ["NEW_DISEASE"])
