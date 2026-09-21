"""Model-free tests for ``openvino convert``'s ONNX-input resolution.

The convert subcommand resolves its ONNX input at ``run()`` time (not via
``type=existing_validation`` at parse time) so it can auto-export:

* ONNX present → returned validated, no export;
* ONNX missing + ``models/yolov8s.pt`` present → exported from the .pt
  first with the export command's defaults, then the conversion proceeds;
* both missing → the standard missing-path error (``PathValidationError``:
  "Path does not exist: ..." + the ARTIFACTS.md hint).

All heavy calls (``export_model``, ``convert_onnx_to_openvino_ir``,
``openvino_conversion_available``) are monkeypatched, so these tests need
no model, GPU, or openvino install.
"""
from types import SimpleNamespace

import pytest

from cli import openvino as cli_openvino
from cli.schema import ExportConfig
from utils import PathValidationError


# ---------------------------------------------------------------------------
# _resolve_convert_onnx — the pure resolution/fallback logic.
# ---------------------------------------------------------------------------
def test_existing_onnx_returned_without_export(tmp_path, monkeypatch):
    onnx = tmp_path / "yolov8s_fp32.onnx"
    onnx.write_bytes(b"x")
    monkeypatch.setattr(cli_openvino, "DEFAULT_MODEL_PT",
                        str(tmp_path / "yolov8s.pt"))
    calls = []
    monkeypatch.setattr(cli_openvino, "export_model",
                        lambda cfg: calls.append(cfg))
    resolved = cli_openvino._resolve_convert_onnx(onnx, imgsz=640)
    assert resolved == onnx.resolve()
    assert calls == []


def test_missing_onnx_and_pt_raises_standard_message(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_openvino, "DEFAULT_MODEL_PT",
                        str(tmp_path / "nope.pt"))
    monkeypatch.setattr(cli_openvino, "DEFAULT_MODEL_FP32",
                        str(tmp_path / "gone.onnx"))
    with pytest.raises(PathValidationError, match="Path does not exist"):
        cli_openvino._resolve_convert_onnx(None, imgsz=640)


def test_missing_onnx_with_pt_exports_first(tmp_path, monkeypatch):
    pt = tmp_path / "yolov8s.pt"
    pt.write_bytes(b"x")
    onnx = tmp_path / "yolov8s_fp32.onnx"
    monkeypatch.setattr(cli_openvino, "DEFAULT_MODEL_PT", str(pt))
    monkeypatch.setattr(cli_openvino, "DEFAULT_MODEL_FP32", str(onnx))
    recorded = []

    def fake_export(cfg):
        recorded.append(cfg)
        onnx.write_bytes(b"onnx")

    monkeypatch.setattr(cli_openvino, "export_model", fake_export)

    resolved = cli_openvino._resolve_convert_onnx(None, imgsz=640)
    assert resolved == onnx.resolve()
    [cfg] = recorded
    assert isinstance(cfg, ExportConfig)
    assert cfg.model == pt
    assert cfg.output == onnx.resolve()
    assert cfg.imgsz == 640
    # Same defaults as `python main.py export` — the auto-export must never
    # be a silent numerical-behavior change relative to an explicit export.
    assert cfg.opset == 17
    assert cfg.dynamic is True
    assert cfg.simplify is True
    assert cfg.device == "cpu"
    assert cfg.nms is False
    assert cfg.validate is True


def test_user_supplied_missing_onnx_also_falls_back(tmp_path, monkeypatch):
    """An explicitly passed missing --model gets the same .pt fallback as
    the default path, and the export honors the requested --imgsz."""
    pt = tmp_path / "yolov8s.pt"
    pt.write_bytes(b"x")
    onnx = tmp_path / "custom_fp32.onnx"
    monkeypatch.setattr(cli_openvino, "DEFAULT_MODEL_PT", str(pt))
    recorded = []
    monkeypatch.setattr(cli_openvino, "export_model",
                        lambda cfg: (recorded.append(cfg),
                                     onnx.write_bytes(b"onnx")))
    resolved = cli_openvino._resolve_convert_onnx(onnx, imgsz=320)
    assert resolved == onnx.resolve()
    assert recorded[0].output == onnx.resolve()
    assert recorded[0].imgsz == 320


# ---------------------------------------------------------------------------
# _run_convert wiring — the auto-exported ONNX actually reaches the IR
# conversion call.
# ---------------------------------------------------------------------------
def test_run_convert_auto_exports_then_converts(tmp_path, monkeypatch):
    pt = tmp_path / "yolov8s.pt"
    pt.write_bytes(b"x")
    onnx = tmp_path / "yolov8s_fp32.onnx"
    out_xml = tmp_path / "yolov8s_openvino.xml"
    monkeypatch.setattr(cli_openvino, "DEFAULT_MODEL_PT", str(pt))
    monkeypatch.setattr(cli_openvino, "DEFAULT_MODEL_FP32", str(onnx))
    monkeypatch.setattr(cli_openvino, "openvino_conversion_available",
                        lambda: True)
    monkeypatch.setattr(cli_openvino, "export_model",
                        lambda cfg: onnx.write_bytes(b"onnx"))
    converted = []
    monkeypatch.setattr(
        cli_openvino, "convert_onnx_to_openvino_ir",
        lambda **kw: (converted.append(kw), str(out_xml))[1],
    )

    args = SimpleNamespace(
        ov_command="convert", model=None, output=out_xml,
        imgsz=640, no_fp16=False,
    )
    cli_openvino.run(args)

    [kw] = converted
    assert kw["onnx_path"] == str(onnx.resolve())
    assert kw["output_xml"] == str(out_xml)
    assert kw["fp16"] is True
