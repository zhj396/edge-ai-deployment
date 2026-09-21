"""ONNX → OpenVINO IR conversion and NNCF INT8 quantization.

Workflow
--------
1. ``convert_onnx_to_openvino_ir`` — IR conversion via ``ovc`` (2023+) or
   the legacy ``mo`` CLI. Outputs FP16 or FP32 IR.
2. ``nncf_quantize_openvino`` — INT8 PTQ using NNCF (Intel's compression
   framework). Same idea as ORT's static PTQ: **FastBiasCorrection** on by
   default; **SmoothQuant** is opt-in and — because NNCF gates it behind
   ``model_type=TRANSFORMER`` — must be explicitly enabled through that
   knob (see the ``smooth_quant`` notes in :func:`nncf_quantize_openvino`).

Both functions use the project's ``CalibrationSampler`` so the same
diverse calibration set is used across ORT INT8 and OpenVINO INT8 —
making them directly comparable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils import get_logger
from . import CalibrationSampler

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from openvino.tools import mo  # legacy Model Optimizer (≤ 2023)
    _MO_AVAILABLE = True
except ImportError:
    mo = None
    _MO_AVAILABLE = False

try:
    import openvino as ov  # provides ov.convert_model in 2023+
    _OVC_AVAILABLE = True
except ImportError:
    ov = None
    _OVC_AVAILABLE = False

try:
    import nncf
    _NNCF_AVAILABLE = True
except ImportError:
    nncf = None
    _NNCF_AVAILABLE = False


def openvino_conversion_available() -> bool:
    return _OVC_AVAILABLE or _MO_AVAILABLE


def nncf_available() -> bool:
    return _NNCF_AVAILABLE


def _resolve_input_shape(onnx_path, imgsz):
    """Build the OpenVINO input shape, mirroring the ONNX's batch dimension.

    The Detect head's DFL Reshape is baked to whatever batch the ONNX was
    exported with — its pattern constant is literally ``[1, 4, 16, 8400]`` for
    a static-batch-1 export (and ``[1, 4, 8400]`` for the second DFL reshape).
    So the IR's batch dim **must match** the ONNX's:

    * forcing ``-1`` (dynamic) on a static-batch-1 ONNX makes the input accept
      batch>1 but the DFL reshape constant still says batch=1 → crash mid-graph
      ("shape of input data (1.64.67200) conflicts with reshape pattern
      (1.4.16.8400)");
    * forcing ``1`` (static) on a dynamic-batch ONNX would reject batch>1 at the
      input even though the graph internals support it.

    We therefore read the ONNX's first input's batch dim and mirror it. Spatial
    dims are pinned to ``imgsz`` so OpenVINO selects fully-shaped oneDNN / VNNI
    kernels regardless of the ONNX's spatial dimensionality.

    Note: the project's ``yolov8s_fp32.onnx`` is the **dynamic** case — input
    is the symbolic dim ``batch`` and the DFL Reshape dims are
    ``[-1, 4, 16, 8400]`` / ``[-1, 4, 8400]`` (batch propagates, not baked to
    1). So the IR is ``[-1, 3, 640, 640]`` and natively runs ``--batch-size 8``
    in one forward — no sub-looping. The static-batch-1 case above guards
    against a *different* (static) export.
    """
    try:
        import onnx
        m = onnx.load(str(onnx_path))
        if m.graph.input:
            dims = m.graph.input[0].type.tensor_type.shape.dim
            if dims:
                bdim = dims[0]
                if bdim.dim_param:  # symbolic name (e.g. "batch") => dynamic
                    b = -1
                elif bdim.dim_value > 0:  # static literal (usually 1)
                    b = int(bdim.dim_value)
                else:  # unspecified => treat as dynamic
                    b = -1
                return [b, 3, imgsz, imgsz]
    except Exception as e:  # pragma: no cover — onnx optional / odd graphs
        logger.warning(
            "Could not read ONNX input shape (%s); defaulting to dynamic "
            "batch [-1, 3, %d, %d]. If the ONNX was exported static-batch, "
            "this IR will crash at the DFL reshape for batch>1.",
            e, imgsz, imgsz,
        )
    return [-1, 3, imgsz, imgsz]


def _save_ir(ov_model, output_path: Path, fp16: bool) -> Optional[bool]:
    """Serialize the IR with an explicit weight-precision choice.

    ``ov.save_model``'s ``compress_to_fp16`` defaults to **True** on the
    pinned build (openvino 2026.3.0 docstring: "Floating point weights are
    compressed to FP16 by default"), so a plain ``save_model`` call silently
    writes FP16 weights even for a ``--no-fp16`` request. Passing the flag
    explicitly in both directions makes the on-disk precision match the
    request on any build that exposes the kwarg. On builds without it
    (TypeError) the IR follows the build default and the log reports
    ``actual_fp16=None`` — unknown — instead of guessing.
    """
    try:
        ov.save_model(ov_model, str(output_path), compress_to_fp16=fp16)
        return fp16
    except TypeError:
        logger.warning(
            "save_model(compress_to_fp16=...) unsupported on this OpenVINO "
            "build; IR weight precision follows the build default "
            "(requested fp16=%s).", fp16,
        )
        ov.save_model(ov_model, str(output_path))
        return None


# ---------------------------------------------------------------------------
# 1. ONNX → OpenVINO IR
# ---------------------------------------------------------------------------
def convert_onnx_to_openvino_ir(
    onnx_path: str,
    output_xml: str,
    fp16: bool = True,
    imgsz: int = 640,
) -> str:
    """Convert an ONNX file to OpenVINO IR (``.xml`` + ``.bin``).

    Parameters
    ----------
    onnx_path
        Source ONNX file.
    output_xml
        Destination ``.xml`` path. The ``.bin`` weights land next to it.
    fp16
        If True, compress weights to FP16 — halves model size, almost
        no accuracy drop on YOLOv8.
    imgsz
        Spatial input size. The IR's batch dimension is **mirrored from the
        ONNX** (static → static, dynamic → dynamic) via :func:`_resolve_input_shape`,
        because the Detect head's DFL Reshape constant is baked to the export
        batch — mismatching it crashes mid-graph. Spatial dims are pinned to
        ``imgsz`` so OpenVINO selects fully-shaped oneDNN / VNNI kernels.
    """
    if not openvino_conversion_available():
        raise ImportError(
            "OpenVINO conversion tools not installed. "
            "Run: pip install -r requirements-openvino.txt"
        )

    p = Path(onnx_path)
    if not p.exists():
        raise FileNotFoundError(f"ONNX file not found: {p}")

    output_path = Path(output_xml)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Mirror the ONNX batch dimension. Forcing -1 on a static-batch-1 ONNX
    # would let the input accept batch>1 while the DFL reshape constant
    # [1,4,16,8400] still says batch=1 → crash at the Detect head.
    # See _resolve_input_shape.
    shape = _resolve_input_shape(p, imgsz)
    logger.info(
        "OpenVINO IR input shape: %s (batch %s)",
        shape, "dynamic" if shape[0] == -1 else f"static {shape[0]}",
    )

    # Preferred API in OpenVINO 2023.1+: ov.convert_model
    if _OVC_AVAILABLE:
        logger.info("Converting via ov.convert_model...")
        ov_model = ov.convert_model(str(p), input=shape)
        # FP16 weight compression happens at save time — see _save_ir for
        # why convert_model carries no FP16 knob and how the *actual*
        # outcome is reported so the log can't claim FP16 it didn't write.
        actual_fp16 = _save_ir(ov_model, output_path, fp16)
    else:  # legacy mo CLI
        logger.info("Converting via legacy mo.convert_model...")
        ov_model = mo.convert_model(
            str(p),
            model_name=output_path.stem,
            input_shape=shape,
            data_type="FP16" if fp16 else "FP32",
            output_dir=str(output_path.parent),
        )
        # mo's data_type= is authoritative — it produces FP16 weights when
        # asked, with no silent FP32 fallback (it raises on a bad dtype).
        actual_fp16 = fp16
        ov.serialize(ov_model, str(output_path))

    size_mb = sum(
        f.stat().st_size for f in output_path.parent.glob(output_path.stem + ".*")
    ) / (1024 * 1024)
    logger.info(
        "OpenVINO IR written: %s (%.2f MB, requested_fp16=%s, actual_fp16=%s)",
        output_path, size_mb, fp16, actual_fp16,
    )
    return str(output_path)


# ---------------------------------------------------------------------------
# 2. NNCF INT8 PTQ
# ---------------------------------------------------------------------------
def nncf_quantize_openvino(
    onnx_path: str,
    output_xml: str,
    data_yaml: str,
    imgsz: int = 640,
    max_samples: int = 300,
    subset_size: int = 64,
    fast_bias_correction: bool = True,
    smooth_quant: bool = False,
    resnet50: Optional[Path] = None,
) -> str:
    """Apply NNCF INT8 post-training quantization.

    Pipeline:
    1. Convert ONNX → FP16 IR via :func:`convert_onnx_to_openvino_ir`.
    2. Use the project's CalibrationSampler for diverse calibration images.
    3. Wrap images in an NNCF ``Dataset`` and call ``nncf.quantize``.
    4. Serialize the INT8 IR to ``output_xml``.

    NNCF algorithm knobs
    --------------------
    * ``smooth_quant=True``  — moves activation outliers into weights
      using a scale-equivalence transform. Helps accuracy when INT8
      crashes from per-channel weight saturation (common on small Conv).
      NOTE: in NNCF 3.x the PTQ pipeline adds the SmoothQuant step **only**
      when ``model_type=TRANSFORMER`` — passing ``smooth_quant_alphas``
      alone is a silent no-op. This function therefore passes
      ``model_type=TRANSFORMER`` + a pinned ``preset=PERFORMANCE`` when the
      flag is on; see the inline comment for the residual confounder
      (batchwise statistics) and the A/B validation requirement.
    * ``fast_bias_correction=True`` — adjusts BN/Conv bias after INT8
      calibration to recover accuracy lost to the quantization bias shift.
    * ``subset_size`` — how many of the sampled calibration images NNCF
      actually runs PTQ over (a subset of ``max_samples``; smaller = faster).
    """
    if not nncf_available():
        raise ImportError(
            "NNCF not installed. Run: pip install -r requirements-openvino.txt"
        )
    if not openvino_conversion_available():
        raise ImportError(
            "OpenVINO conversion tools required for NNCF quantization."
        )

    onnx_path = Path(onnx_path)
    output_path = Path(output_xml)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Base FP16 IR ------------------------------------------------
    fp16_xml = output_path.with_name(output_path.stem + "_fp16.xml")
    if not fp16_xml.exists():
        logger.info("Building FP16 IR first...")
        convert_onnx_to_openvino_ir(
            str(onnx_path), str(fp16_xml), fp16=True, imgsz=imgsz,
        )

    # --- 2. Calibration set --------------------------------------------
    sampler = CalibrationSampler(
        data_yaml=data_yaml,
        calibration_size=max_samples,
        local_weights=resnet50,
    )
    calibration_paths = sampler.sample()
    logger.info("NNCF calibration images: %d", len(calibration_paths))

    from .preprocess import preprocess_imgs  # late import to avoid cycle

    class _YOLODataset:
        """Feed preprocessed letterboxed/normalized batches to NNCF.

        ``__getitem__`` returns a ``[1, 3, imgsz, imgsz]`` float32 numpy batch
        already routed through :func:`preprocess_imgs`, so the letterbox +
        normalization NNCF sees at calibration time is byte-identical to the
        deployed ``OpenVINOEngine.infer`` path. No separate ``transform_fn``
        is needed — we preprocess at fetch time rather than after.
        """

        def __init__(self, paths, imgsz):
            self.paths = paths
            self.imgsz = imgsz

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            batch = preprocess_imgs(
                [self.paths[idx]],
                imgsz=self.imgsz,
                device="cpu",
                original=False,
            )
            return batch["images"].cpu().numpy()

    # --- 3. Load + quantize --------------------------------------------
    ov_model = ov.Core().read_model(str(fp16_xml))
    calibration_dataset = nncf.Dataset(
        _YOLODataset(calibration_paths, imgsz=imgsz),
    )

    # Head exclusion — same rationale AND (on name-preserving builds) the
    # same boundary as the ORT path's ``HEAD_NAME_PREFIXES=["/model.22/"]``
    # (src/quantize.py): the Detect head's Sigmoid is the conf cliff and
    # must stay high-precision or cls scores collapse.
    #
    # Two-layer IgnoredScope:
    #   patterns=[r"/model\.22/.*"] — the exact Detect subgraph on builds
    #       that keep ONNX friendly names. Verified on the pinned openvino
    #       2026.3.0 + this repo's yolov8s_fp32.onnx: 135/504 graph nodes
    #       carry the /model.22/ prefix (incl. the head's Conv branches,
    #       which an op-type scope cannot express).
    #   types=["Sigmoid", "Softmax"] — portable floor for builds that
    #       rewrote friendly names. YOLOv8s carries exactly one Sigmoid and
    #       one Softmax, both inside the head (verified on the same IR), so
    #       this layer is head-precise, not a coarse superset.
    # validate=False: unmatched patterns/types are skipped instead of
    # raising, so the scope stays portable across IR op-sets. To keep the
    # name layer's portability failure LOUD instead of silent, resolution
    # is checked against the actual graph and logged below.
    head_nodes = [
        op.get_friendly_name() for op in ov_model.get_ops()
        if op.get_friendly_name().startswith("/model.22/")
    ]
    logger.info(
        "Head exclusion: %d graph nodes match /model.22/ by name "
        "(pattern layer %s)",
        len(head_nodes), "active" if head_nodes else "INERT",
    )
    if not head_nodes:
        logger.warning(
            "No nodes matched /model.22/ — this OpenVINO build rewrote the "
            "ONNX friendly names. Name-pattern exclusion is inert; the head "
            "stays protected only by the op-type layer (Sigmoid/Softmax), "
            "which does NOT cover the head's Conv branches. Compare "
            "consistency metrics against the ORT INT8 path before trusting "
            "this IR."
        )
    ignored_scope = nncf.IgnoredScope(
        patterns=[r"/model\.22/.*"],
        types=["Sigmoid", "Softmax"],
        validate=False,
    )

    # SmoothQuant gating (root-caused on the pinned nncf 3.3.0): the PTQ
    # pipeline adds the SmoothQuant step ONLY when model_type ==
    # ModelType.TRANSFORMER (nncf/quantization/algorithms/post_training/
    # pipeline.py) — passing smooth_quant_alphas alone, with the default
    # model_type=None, is a SILENT NO-OP. Empirically verified:
    #   model_type=None        -> [MinMaxQuantization, FastBiasCorrection]
    #   model_type=TRANSFORMER -> [SmoothQuant], [MinMax, FastBiasCorrection]
    # So the flag must also switch model_type. Two side effects of
    # TRANSFORMER, one controlled and one disclosed:
    #   - preset default flips to MIXED (asymmetric activations). We pin
    #     preset=PERFORMANCE explicitly so a flag-on-vs-off A/B stays
    #     attributable to SmoothQuant alone.
    #   - batchwise statistics are auto-disabled
    #     (nncf/openvino/quantization/quantize_model.py). Not controllable
    #     via preset; validate the NET effect with a consistency run
    #     (INT8 vs FP16 baseline, flag on vs off) before trusting the flag.
    sq_kwargs: dict = {}
    if smooth_quant:
        sq_kwargs = {
            "model_type": nncf.ModelType.TRANSFORMER,
            "preset": nncf.QuantizationPreset.PERFORMANCE,
        }
    sq_alphas = (
        nncf.AdvancedSmoothQuantParameters(matmul=0.95, convolution=0.95)
        if smooth_quant else None
    )
    # NNCF 3.x API (requirements-openvino.txt pins nncf==3.3.0). Two knobs
    # were renamed vs NNCF 2.x, which is why the pin matters:
    #   - bias correction moved OUT of advanced_parameters.weights_bias_correction
    #     to a TOP-LEVEL ``fast_bias_correction`` kwarg on nncf.quantize;
    #   - IgnoredScope renamed ``op_types`` -> ``types``.
    # The SmoothQuant model_type gate above is likewise a 3.3.0-verified
    # behavior; tests/test_openvino_optional_imports.py guards it.
    # A 2.x install would TypeError on the renamed knobs; the pin sidesteps that.
    logger.info(
        "Running NNCF %s INT8 PTQ (smooth_quant=%s%s, bias_correction=%s)",
        nncf.__version__, smooth_quant,
        ", model_type=transformer, preset=performance" if smooth_quant else "",
        fast_bias_correction,
    )
    quantized_model = nncf.quantize(
        ov_model, calibration_dataset,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            smooth_quant_alphas=sq_alphas,
        ),
        ignored_scope=ignored_scope,
        **sq_kwargs,
    )

    ov.save_model(quantized_model, str(output_path))
    size_mb = sum(
        f.stat().st_size for f in output_path.parent.glob(output_path.stem + ".*")
    ) / (1024 * 1024)
    logger.info("NNCF INT8 IR written: %s (%.2f MB)", output_path, size_mb)
    return str(output_path)
