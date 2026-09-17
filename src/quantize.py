"""Static INT8 post-training quantization for YOLOv8s ONNX models.

Quantization policy
-------------------
* **Format**: QDQ (Quantize-Dequantize pairs) — better ONNX Runtime kernel fusion than the legacy
QOperator format.
* **Activations**: QUInt8 (asymmetric, unsigned) — the default for AVX-512 VNNI / VNNI4
and ARM NEON dot-product paths.
* **Weights**: QInt8 (symmetric, signed) — symmetric weights are standard and well-supported by
x86/ARM GEMM kernels.
* **Per-channel** weight quantization — preserves accuracy on conv layers whose channel-wise
dynamic range differs sharply.
* **Calibration**: MinMax (fast) or Entropy (KL-divergence, better for tight distributions).
* **Head protection**: the entire YOLOv8 Detect head (``/model.22/...``) is kept in FP32.
Per-channel INT8 QDQ on the classification-logit branch collapses the post-Sigmoid class scores to
zero (empirically: all-0 detections) — the cls-logit distribution is too tight/sensitive for
asymmetric uint8 with a MinMax scale distorted by negative outliers, and the box-decoder branch
survives quantization but the cls branch does not. The head is <5% of FLOPs, so keeping it FP32
costs negligible speed and is decisive for accuracy. Two residual layers guard the Sigmoid/Softmax
nodes inside the head as belt-and-suspenders: (1) name-based full node exclusion resolved from the
pre-processed graph, and (2) op-type output-exclusion as a fallback.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import List, Optional

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
# convs, cv3 cls convs, DFL, Sigmoid, Concat). Excluding every node in that
# subgraph from quantization is the primary protection: it keeps the cls-logit
# branch (cv3 → Concat → Sigmoid) in FP32, which is where per-channel INT8 QDQ
# otherwise collapses class scores to zero. Resolved to concrete node names from
# the pre-processed graph at runtime — see ``_resolve_node_names_by_prefix``.
#
# Two residual op-type layers guard the Sigmoid/Softmax nodes inside the head as a fallback (they
# become no-ops once the prefix exclusion covers the head, but stay cheap insurance against a
# renamed/merged head node in a future opset).

HEAD_NAME_PREFIXES = ["/model.22/"]  # YOLOv8 Detect head — whole-subgraph FP32

OP_TYPES_TO_EXCLUDE_OUTPUT_QUANTIZATION = [
    "Sigmoid",  # head objectness/confidence
    "Softmax",  # DFL probability distribution
]

# Op types whose nodes are skipped entirely (input + weight + output). Resolved to node names at
# runtime from the pre-processed graph — see ``_resolve_node_names_by_op_type``.

NODE_TYPES_TO_EXCLUDE = [
    "Sigmoid",  # head activations
    "Softmax",  # DFL softmax
]


def _resolve_node_names_by_op_type(
    model_path: Path, op_types: List[str]
) -> List[str]:
    """Resolve concrete node names for ``op_types`` from the graph to be quantized.

    ``quantize_static(nodes_to_exclude=...)`` matches by ``node.name``, not by op type, so the
    op-type policy must be translated into names from the very graph quantization will see — i.e.
    the *pre-processed* model, since ``quant_pre_process`` may rename/restructure nodes. If
    preprocessing failed, resolution falls back to the original FP32 graph.

    Unnamed nodes (empty ``node.name``) are skipped with a warning: name-based exclusion can't
    target them, and including ``""`` would wrongly match *every* unnamed node. They fall back to
    ``OpTypesToExcludeOutputQuantization``.
    """
    targets = set(op_types)
    resolved: List[str] = []
    unnamed = 0
    try:
        model = onnx.load_model(str(model_path), load_external_data=False)
    except Exception as e:  # pragma: no cover - best-effort resolution
        logger.warning(
            "Could not load %s to resolve excluded node names (%s); "
            "falling back to op-type output exclusion only.",
            model_path, e,
        )
        return []

    for node in model.graph.node:
        if node.op_type in targets:
            if node.name:
                resolved.append(node.name)
            else:
                unnamed += 1

    if unnamed:
        logger.warning(
            "Found %d unnamed node(s) matching %s in %s; cannot exclude by "
            "name — relying on OpTypesToExcludeOutputQuantization instead.",
            unnamed, sorted(targets), model_path.name,
        )
    logger.info(
        "Resolved %d node(s) to exclude by name (op types=%s).",
        len(resolved), sorted(targets),
    )
    return resolved


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

    Uses ``PIL.Image.verify()`` (header/structure check, *no* decode), so cost on a typical
    300-image calibration set is <100 ms. Goal here is to keep ``__len__`` honest — without
    filtering, a mid-run batch failure skips a ``batch_size`` window of ``self.index`` and the
    reported batch count diverges from the number of iterations actually delivered.

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

    Constructor pre-filters unreadable images via :func:`_filter_readable`, so ``__len__`` reports
    the actual number of iterations the reader will yield. Mid-run transient failures (rare since
    pixel decode happens here) still skip with an explicit log; pre-filtering keeps the budget
    honest.
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
    device: str = "cpu",
    extra_options: Optional[dict] = None,
) -> str:
    """Run static INT8 quantization and write the result to ``onnx_int8_path``.

    Steps
    -----
    1. ``quant_pre_process`` — ONNX shape inference + baseline graph optimization.
    2. Build the calibration set (``CalibrationSampler`` + ``YOLOv8CalibrationDataReader``).
    3. Determine the session input name from the pre-processed graph.
    4. Resolve excluded node names (Detect head by prefix + Sigmoid/Softmax op types).
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
    # Read the input name straight from the ONNX graph — no InferenceSession needed:
    # spinning one up (and, on a CUDA build, the arena) just to read metadata costs
    # dearly, while ``graph.input[0].name`` is the same value for ~free.
    # Fall back to a CPU session only if the graph can't be parsed (e.g. an externally-
    # rewritten model whose input lives in external data we chose not to load).

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
    # Whole-Detect-head exclusion by prefix is primary (keeps the cls-logit branch in FP32);
    # op-type exclusion of Sigmoid/Softmax is residual insurance.
    # Both resolved from the pre-processed graph — the one quantize_static sees — since
    # quant_pre_process may rename/restructure nodes.

    excluded_node_names = sorted(set(
        _resolve_node_names_by_prefix(model_for_quant, HEAD_NAME_PREFIXES)
        + _resolve_node_names_by_op_type(model_for_quant, NODE_TYPES_TO_EXCLUDE)
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

    # Single source of truth for provider selection — same helper the engine, benchmark, and
    # consistency use, so CUDA calibration (device="cuda") runs on the same tuned CUDA EP as
    # deployment. Note: ``sampler.device``, not the raw ``device`` argument — CalibrationSampler
    # already downgraded "cuda" to "cpu" when CUDA is unavailable, and passing the raw value here
    # would send ORT's calibration forward to a CUDA EP that doesn't exist on this machine while
    # the feature extraction silently ran on CPU (asymmetric degradation).

    providers = select_providers(sampler.device)

    quantize_static(
        model_input=str(model_for_quant),
        model_output=str(int8_path),
        calibration_data_reader=calibrator,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=calib_method,
        calibration_providers=providers,
        per_channel=True,
        op_types_to_quantize=None,                 # quantize all eligible
        nodes_to_exclude=excluded_node_names,      # Detect head prefix + Sigmoid/Softmax ops
        extra_options=base_extra,
    )

    # ---- 6. cleanup ----
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
