# tests/test_onnx_parity.py
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
pytest.importorskip("onnxruntime")


def test_tiny_mobilenet_export_parity(tmp_path):
    from octscreen.optimize.export_onnx import _export, parity_check
    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=4).eval()
    spec = {"image_size": [64, 64]}
    out = tmp_path / "m.onnx"
    _export(model, torch.zeros(1, 3, 64, 64), out, 17, True, "auto")
    assert parity_check(model, out, spec, n=2) < 1e-3