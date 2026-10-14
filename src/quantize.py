"""Static INT8 post-training quantization for YOLOv8s ONNX models.

Quantization policy
-------------------
* **Format**: QDQ (Quantize-Dequantize pairs) — the format ONNX Runtime fuses into INT8 kernels.
* **Activations**: QUInt8 (asymmetric, unsigned) — with QInt8 weights this gives the u8s8 GEMM
type that AVX-512 VNNI and ARM NEON dot-product kernels consume.
* **Weights**: QInt8 (symmetric, signed) — symmetric weights are standard and well-supported by
x86/ARM GEMM kernels.
* **Per-channel** weight quantization — preserves accuracy on conv layers whose channel-wise
dynamic range differs sharply.
* **Calibration**: MinMax or Entropy (KL-divergence).
* **Head protection**: the entire YOLOv8 Detect head (``/model.22/...``) is kept in FP32;
quantizing the cls-logit branch reduces the post-Sigmoid class scores to zero in the validated
YOLOv8s configuration. Two exclusion layers apply: (1) name-based full node exclusion resolved
from the pre-processed graph, and (2) op-type output-exclusion as a fallback.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)
from PIL import Image

from utils import get_logger, select_providers
from . import CalibrationSampler, preprocess_imgs

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Head-protection policy — keep the entire Detect head in FP32
# ---------------------------------------------------------------------------
# The whole YOLOv8 Detect head lives under the ``/model.22/`` name prefix (cv2 box
# convs, cv3 cls convs, DFL, Sigmoid, Concat). Every node in that subgraph is
# excluded from quantization. Resolved to concrete node names from
# the pre-processed graph at runtime — see ``_resolve_node_names_by_prefix``.

HEAD_NAME_PREFIXES = ["/model.22/"]  # YOLOv8 Detect head — whole-subgraph FP32

# Output-quantization exclusion applies graph-wide, so it is the layer that
# still protects the head when name-based resolution finds nothing (renamed
# head nodes); in a YOLOv8 export it also covers the backbone/neck Sigmoid
# outputs produced by SiLU decomposition.

OP_TYPES_TO_EXCLUDE_OUTPUT_QUANTIZATION = [
    "Sigmoid",  # head cls activations
    "Softmax",  # DFL probability distribution
]


def _resolve_node_names_by_prefix(
    model_path: Path, prefixes: List[str]
) -> List[str]:
    """Resolve concrete node names whose name starts with any of ``prefixes``.

    Used to exclude the whole YOLOv8 Detect head (``/model.22/``) from quantization by name
    prefix — a subgraph-level skip that op-type resolution can't express (the head is mostly
    Conv/Mul/Concat, indistinguishable from the backbone by op type). Resolved from the model
    that will actually be quantized — normally the ``quant_pre_process`` output, so the names
    match what ``quantize_static`` sees; if preprocessing failed, the original FP32 graph
    (node names may differ in that fallback).
    """
    if not prefixes:
        return []
    resolved: List[str] = []
    try:
        model = onnx.load_model(str(model_path), load_external_data=False)
    except Exception as e:  # pragma: no cover - best-effort resolution
        logger.warning(
            "Could not load %s to resolve excluded node names by prefix (%s).",
            model_path, e,
        )
        return []
    for node in model.graph.node:
        if node.name and any(node.name.startswith(p) for p in prefixes):
            resolved.append(node.name)
    logger.info(
        "Resolved %d node(s) to exclude by prefix (%s).",
        len(resolved), prefixes,
    )
    return resolved


# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------
def _filter_readable(paths: List[Path]) -> tuple[List[Path], list]:
    """Cheap pre-filter: keep paths whose bytes will decode as an image.

    Uses ``PIL.Image.verify()`` (header/structure check, *no* decode), so the per-image cost
    is a header read. This keeps ``__len__`` aligned with the number of batches the reader will
    actually deliver.

    Returns (kept_paths, dropped_with_reasons). ``dropped_with_reasons`` is a list of (Path, str) so
    the caller can log a sample without re-opening.
    """
    good: List[Path] = []
    bad: List = []
    for p in paths:
        try:
            if not p.exists() or not p.is_file():
                bad.append((p, f"not a regular file: {p}"))
                continue
            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            bad.append((p, str(e)))
        else:
            good.append(p)
    return good, bad


class YOLOv8CalibrationDataReader(CalibrationDataReader):
    """ONNX Runtime calibration data reader backed by ``preprocess_imgs``.

    Constructor pre-filters unreadable images via :func:`_filter_readable` (a header/structure
    check), so ``__len__`` reports the number of batches the reader will yield. Images that
    pass the header check but fail during pixel decode are skipped mid-run with an explicit
    log; ``__len__`` then over-counts.
    """

    def __init__(
        self,
        calibration_imgs: List[Path],
        imgsz: int = 640,
        batch_size: int = 1,
        num_workers: int = 4,
        input_name: str = "images",
    ) -> None:
        raw = list(calibration_imgs)
        readable, dropped = _filter_readable(raw)
        if dropped:
            logger.warning(
                "Calibration reader: %d/%d images unreadable, dropped pre-run",
                len(dropped), len(raw),
            )
            for p, why in dropped[:5]:
                logger.warning("  - %s: %s", p, why)
            if len(dropped) > 5:
                logger.warning("  - … and %d more", len(dropped) - 5)
        if not readable:
            raise RuntimeError(
                f"No readable calibration images out of {len(raw)} supplied"
            )

        self.calibration_imgs = readable
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_name = input_name
        self.index = 0

        logger.info(
            "Calibration reader init | images=%d | batch=%d | imgsz=%d",
            len(self.calibration_imgs), batch_size, imgsz,
        )

    def get_next(self) -> Optional[dict]:
        while self.index < len(self.calibration_imgs):
            batch_paths = self.calibration_imgs[self.index : self.index + self.batch_size]
            try:
                batch_data = preprocess_imgs(
                    img_paths=batch_paths,
                    imgsz=self.imgsz,
                    device="cpu",
                    fp16=False,
                    original=False,
                    num_workers=self.num_workers,
                )
                if batch_data.get("images") is None:
                    logger.warning(
                        "Batch preprocessing returned empty data: %s",
                        batch_paths,
                    )
                    self.index += self.batch_size
                    continue
                images_np = batch_data["images"].cpu().numpy()
                if images_np.dtype != np.float32:
                    images_np = images_np.astype(np.float32)
                self.index += self.batch_size
                return {self.input_name: images_np}
            except Exception as e:
                logger.error("Calibration batch failed %s: %s", batch_paths, e)
                self.index += self.batch_size
                continue

        logger.info("Calibration data reading complete")
        return None

    def reset(self) -> None:
        self.index = 0
        gc.collect()

    def __len__(self) -> int:
        # Number of *batches* the reader yields, not the raw image count — ORT's calibration loop
        # may use __len__ for progress reporting, and reporting images when batch_size>1 over-counts
        # the iterations.

        return (len(self.calibration_imgs) + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def quantize_onnx_to_int8(
    onnx_fp32_path: str,
    onnx_int8_path: str,
    calibration_imgs: str,
    imgsz: int = 640,
    batch_size: int = 1,
    max_samples: int = 300,
    method: str = "MinMax",
    resnet50: Optional[Path] = None,
    device: Union[str, int] = "cpu",
    extra_options: Optional[dict] = None,
    activation_type: QuantType = QuantType.QUInt8,
) -> str:
    """Run static INT8 quantization and write the result to ``onnx_int8_path``.

    ``activation_type`` defaults to ``QUInt8`` (asymmetric, unsigned). Pass
    ``QInt8`` for a TensorRT-target QDQ model, together with
    ``extra_options={"QuantizeBias": False}`` (TRT rejects UINT8 zero-points and
    the INT32 bias DequantizeLinear; bias stays FP32). The head-exclusion scope
    (``HEAD_NAME_PREFIXES``) is identical for both.

    Steps
    -----
    1. ``quant_pre_process`` — ONNX shape inference + baseline graph optimization.
    2. Build the calibration set (``CalibrationSampler`` + ``YOLOv8CalibrationDataReader``).
    3. Determine the session input name from the pre-processed graph.
    4. Resolve excluded node names (Detect head by name prefix).
    5. ``quantize_static`` — QDQ insertion with the exclusion lists and extra options.
    6. Cleanup the temp pre-processed model.
    """
    fp32_path = Path(onnx_fp32_path)
    int8_path = Path(onnx_int8_path)

    if not fp32_path.exists():
        raise FileNotFoundError(f"Model not found: {fp32_path}")

    # Validate the calibration method up front, before the expensive preprocess/sample stages:
    # only 'MinMax' and 'Entropy' map to a CalibrationMethod; anything else is a hard error.
    if method == "MinMax":
        calib_method = CalibrationMethod.MinMax
    elif method == "Entropy":
        calib_method = CalibrationMethod.Entropy
    else:
        raise ValueError(
            f"Unknown calibration method: {method!r} (expected 'MinMax' or 'Entropy')"
        )

    # ---- 1. quant_pre_process ----
    preprocessed_path = fp32_path.with_name(fp32_path.stem + "_preprocessed.onnx")

    try:
        logger.info("Running ONNX quant-preprocess...")
        quant_pre_process(
            input_model=str(fp32_path),
            output_model_path=str(preprocessed_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=True,
        )
        model_for_quant = preprocessed_path
        logger.info("Preprocess complete")
    except Exception as e:
        logger.warning("Preprocess failed (%s); using original model", e)
        model_for_quant = fp32_path

    # ---- 2. calibration set ----
    data_yaml = Path(calibration_imgs) / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    logger.info(
        "Starting INT8 quantization (batch=%d, max_samples=%d, method=%s)",
        batch_size, max_samples, method,
    )

    sampler = CalibrationSampler(
        data_yaml=data_yaml,
        calibration_size=max_samples,
        local_weights=resnet50,
        device=device,
    )
    calibration_imgs_list = sampler.sample()

    # ---- 3. determine input name from model ----
    # Read ``graph.input[0].name``; if the model cannot be loaded as a graph,
    # fall back to a CPU InferenceSession to read the input name.

    try:
        graph_model = onnx.load_model(
            str(model_for_quant), load_external_data=False
        )
        input_name = graph_model.graph.input[0].name
    except Exception as e:
        logger.warning(
            "Could not read input name from graph (%s); falling back to an InferenceSession.",
            e,
        )
        session = ort.InferenceSession(str(model_for_quant), providers=select_providers("cpu"))
        input_name = session.get_inputs()[0].name
        del session

    calibrator = YOLOv8CalibrationDataReader(
        calibration_imgs=calibration_imgs_list,
        imgsz=imgsz,
        batch_size=batch_size,
        num_workers=4,
        input_name=input_name,
    )

    # ---- 4. resolve head nodes to exclude by name ----
    # Excluded node names are resolved from ``model_for_quant`` so they match the
    # graph ``quantize_static`` receives (quant_pre_process may rename nodes).

    excluded_node_names = sorted(set(
        _resolve_node_names_by_prefix(model_for_quant, HEAD_NAME_PREFIXES)
    ))

    # ---- 5. extra_options ----
    base_extra = {
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
        "EnableSubgraph": True,
        "OpTypesToExcludeOutputQuantization": OP_TYPES_TO_EXCLUDE_OUTPUT_QUANTIZATION,
    }
    if extra_options:
        base_extra.update(extra_options)

    # ``providers = select_providers(sampler.device)`` — calibration runs on the
    # same device the sampler used.

    providers = select_providers(sampler.device)

    try:
        quantize_static(
            model_input=str(model_for_quant),
            model_output=str(int8_path),
            calibration_data_reader=calibrator,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=QuantType.QInt8,
            calibrate_method=calib_method,
            calibration_providers=providers,
            per_channel=True,
            op_types_to_quantize=None,                 # quantize all eligible
            nodes_to_exclude=excluded_node_names,      # Detect head, by name prefix
            extra_options=base_extra,
        )
    finally:
        # ---- cleanup: remove the temp preprocessed model ----
        if model_for_quant != fp32_path:
            try:
                Path(model_for_quant).unlink(missing_ok=True)
            except Exception:
                pass

    size_mb = int8_path.stat().st_size / (1024 * 1024)
    logger.info(
        "INT8 quantization complete | size=%.2f MB | path=%s",
        size_mb, int8_path,
    )
    return str(int8_path)
