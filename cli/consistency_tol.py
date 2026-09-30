"""Pure tolerance-selection logic for the ``consistency`` subcommand.

Lives outside ``cli/consistency.py`` so it can be unit-tested without importing
``src`` (which pulls ultralytics). The CLI module imports these names from here.

Why this is a function of *which models* are being compared, not a fixed default:
the old fixed ``1e-3 / 1e-2`` default let a ~1e-3 ORT-CUDA-EP-vs-PyTorch-CUDA
divergence PASS in tensor mode for PT↔FP32 — a false pass, because that
divergence lands on the ``conf=0.25`` cliff and inflates FP32-CUDA ``infer``
detections ~30×. Reduced-precision models (INT8 quant OR FP16) genuinely need
the loose pair: INT8 because quantization noise raises advisory p99 via
low-confidence boxes NMS drops (CLAUDE.md invariant 12); FP16 because ~3-digit
mantissa precision on the YOLOv8 Detect head's unbounded box-decode accumulation
diverges to the single digits on a 0–640 coordinate scale — strict allclose
cannot pass and should not. The conf-cliff gate remains the authoritative
catcher in both cases; the loose pair only relaxes the advisory tensor-mode
allclose so the run isn't a blanket 100% FAIL.
"""
from __future__ import annotations

from pathlib import Path

# PT↔FP32: tight enough that a cliff-amplifying ~1e-3 cross-EP divergence FAILS
# tensor-mode allclose instead of false-passing.
STRICT_ATOL = 1e-4
STRICT_RTOL = 1e-3
# Any side INT8: quantization noise needs the headroom (advisory tensor stats only;
# the per-image detection gate is authoritative — see utils/comparison.compare_tensors).
LOOSE_ATOL = 1e-3
LOOSE_RTOL = 1e-2


def _stem(model) -> str:
    """Lower-cased stem of a model spec, tolerating backend prefixes.

    ``tensorrt:models/yolov8s_fp16.engine`` and ``ort_cpp:.../x_int8.onnx`` still
    resolve to ``yolov8s_fp16`` / ``x_int8`` because :py:class:`Path` treats the
    ``<tag>:`` prefix as a drive letter and filename parsing proceeds normally.
    """
    return Path(str(model)).stem.lower()


def involves_int8(model1, model2) -> bool:
    """True if either model path looks like the INT8 quantized model.

    Path-stem based (``yolov8s_int8.onnx`` → True) so it works on the documented
    defaults and on user-supplied paths without loading the model. A non-reduced
    comparison (PT↔FP32) gets the strict pair; any INT8 side keeps the loose pair.
    """
    s1, s2 = _stem(model1), _stem(model2)
    return "int8" in s1 or "int8" in s2


def involves_fp16(model1, model2) -> bool:
    """True if either model path looks like an FP16 model.

    Path-stem based (``yolov8s_fp16.engine`` → True, ``yolov8s_fp32.onnx`` → False)
    so it works on the TRT FP16 engine and any future FP16 IR/ONNX without loading
    the model. FP16-vs-FP32 raw-tensor divergence cannot meet strict allclose —
    see module docstring — so an FP16 side selects the loose pair; the conf-cliff
    gate (``utils/comparison.compare_tensors``) stays authoritative.
    """
    s1, s2 = _stem(model1), _stem(model2)
    return "fp16" in s1 or "fp16" in s2


def needs_loose_pair(model1, model2) -> bool:
    """True if either side is reduced-precision (INT8 quant or FP16)."""
    return involves_int8(model1, model2) or involves_fp16(model1, model2)


def resolve_tolerances(atol, rtol, model1, model2):
    """Pick (atol, rtol): explicit values win; else auto by reduced-precision side.

    ``atol``/``rtol`` come straight from argparse (``None`` when the user didn't
    pass them). Passing only one is rejected — mixing a strict atol with a loose
    rtol (or vice-versa) is almost certainly a mistake.
    """
    if atol is not None and rtol is not None:
        return atol, rtol
    if atol is not None or rtol is not None:
        raise SystemExit(
            "Specify both --atol and --rtol, or neither (auto: strict for "
            "PT↔FP32, loose when INT8 or FP16 is involved). Got only one."
        )
    if needs_loose_pair(model1, model2):
        return LOOSE_ATOL, LOOSE_RTOL
    return STRICT_ATOL, STRICT_RTOL
