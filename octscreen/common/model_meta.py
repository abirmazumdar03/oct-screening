# octscreen/common/model_meta.py
"""Load and validate model_meta.json (contract C5). Dependency-free: json + pathlib only."""
import json
from pathlib import Path

SCHEMA_VERSION = 1
REQUIRED_TOP = ["schema_version", "package_name", "package_version", "class_names", "preprocess", "classifier"]
REQUIRED_PRE = ["image_size", "resize_mode", "color", "mean", "std"]
REQUIRED_CLS = ["format", "file", "input_name", "output_name", "output_type", "precision", "temperature"]


def validate_meta(meta, package_dir=None):
    """Return a list of problems (empty list = valid)."""
    errors = [f"missing key: {k}" for k in REQUIRED_TOP if k not in meta]
    if errors:
        return errors
    if meta["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version {meta['schema_version']} != supported {SCHEMA_VERSION}")
    names = meta["class_names"]
    if not names or len(set(names)) != len(names):
        errors.append("class_names must be non-empty and unique")
    errors += [f"missing preprocess.{k}" for k in REQUIRED_PRE if k not in meta["preprocess"]]
    cls = meta["classifier"]
    errors += [f"missing classifier.{k}" for k in REQUIRED_CLS if k not in cls]
    if cls.get("output_type") not in ("logits", "probabilities"):
        errors.append("classifier.output_type must be 'logits' or 'probabilities'")
    t = cls.get("temperature")
    if not isinstance(t, (int, float)) or t <= 0:
        errors.append("classifier.temperature must be a positive number")
    ex = meta.get("explainer")
    if ex is not None:
        errors += [f"missing explainer.{k}" for k in ("type", "file") if k not in ex]
    if package_dir is not None:
        p = Path(package_dir)
        if "file" in cls and not (p / cls["file"]).is_file():
            errors.append(f"classifier file not found: {p / cls['file']}")
        if ex and "file" in ex and not (p / ex["file"]).is_file():
            errors.append(f"explainer file not found: {p / ex['file']}")
    return errors


def load_meta(package_dir):
    """Read and validate <package_dir>/model_meta.json. Raises ValueError with all problems listed."""
    path = Path(package_dir) / "model_meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_meta(meta, package_dir)
    if errors:
        raise ValueError(f"Invalid model package {package_dir}:\n - " + "\n - ".join(errors))
    return meta