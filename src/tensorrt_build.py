"""TensorRT engine build + Ultralytics native export for YOLOv8s.

Two build paths live here (the third, ``trtexec``, is the shell script
``scripts/build_trt_engines.sh``):

* **Path A — Python API** (:func:`build_tensorrt_engine`): ``.onnx -> .engine``
  via ``trt.Builder`` + ``OnnxParser`` + ``OptimizationProfile`` +
  ``BuilderFlag.FP16``/``INT8`` + ``IInt8EntropyCalibrator2``. This is the
  primary, full-control path and the strong interview story (the programmatic
  TRT-10 API: ``set_memory_pool_limit``, ``build_serialized_network``, the
  tensor-name inference API exercised in :mod:`src.tensorrt_engine`).

* **Path B — Ultralytics native** (:func:`export_tensorrt_ultralytics`):
  ``.pt -> .engine`` via ``model.export(format="engine")``. Recommended for
  Kaggle / Colab where ``trtexec`` is unavailable — Ultralytics invokes the
  TRT Python API internally in one step. Less control over profiles/workspace
  but zero extra moving parts.

Both paths produce the same raw ``[B, 4+nc, 8400]`` head that
:class:`src.tensorrt_engine.TensorRTEngine` consumes. INT8 calibration reuses
the **shared** :func:`src.preprocess.preprocess_single` (letterbox + BGR->RGB +
CHW + /255) so the calibrator's numerics exactly match the deployed path, and
the calibration image list is built by the **shared**
:class:`src.sampler.CalibrationSampler` — no TRT-specific sampler fork.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from utils import get_logger
from .preprocess import preprocess_single
# Reuse the engine module's guarded ``trt`` / ``cuda`` imports + the lean-bindings
# helpers — one CUDA dialect end-to-end, one availability predicate.
from .tensorrt_engine import (
    _cuda_call,
    _cuda_check,
    cuda,
    set_cuda_visible_devices,
    tensorrt_available,
    trt,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Build constants — the only TRT-build-specific magic numbers. The inference
# engine's buffer-sizing constants (YOLOV8_BOX_OFFSET / COCO_MAX_CLASSES) live
# in :mod:`src.tensorrt_engine`; this module only needs the workspace + the
# box-count formula.
# ---------------------------------------------------------------------------
WORKSPACE_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB — comfortable on a T4 (16 GiB)


def expected_num_boxes(img_size: int = 640) -> int:
    """Raw YOLOv8 candidate count for a square input (3 strides: 8/16/32).

    ``(img/8)^2 + (img/16)^2 + (img/32)^2`` — 8400 for 640x640.
    """
    if img_size % 32 != 0:
        raise ValueError(
            f"img_size must be divisible by 32 (got {img_size}); "
            "YOLOv8 detection strides are 8/16/32."
        )
    return (img_size // 8) ** 2 + (img_size // 16) ** 2 + (img_size // 32) ** 2


def trt_build_available() -> bool:
    """True iff the build path can run (tensorrt + cuda-python installed)."""
    return tensorrt_available()


# ---------------------------------------------------------------------------
# Detect-head exclusion — the TRT Path-A equivalent of ORT invariant #3
# (``src/quantize.py::HEAD_NAME_PREFIXES``). The whole YOLOv8 Detect head lives
# under the ``/model.22/`` ONNX name prefix (cv2 box convs, cv3 cls convs, DFL,
# Sigmoid, Concat, box-decode arithmetic). Quantizing that subgraph to INT8
# collapses the head's unbounded box-decode accumulation (~256 levels, no
# exponent range) → catastrophic box-coord divergence + conf-cliff flips. T4
# proof (2026-08-28): whole-net INT8 → max_diff 367–593, ``detection_fail_rate``
# 50%; FP16-whole-net survives (max_diff ~15, FP16 has exponent range), so head
# FP32 protection is **mandatory for INT8, optional for FP16**. Pinning the head
# to FP32 under an INT8 build mirrors the ORT QDQ path's whole-head FP32 skip.
HEAD_NAME_PREFIXES = ["/model.22/"]  # YOLOv8 Detect head — whole-subgraph FP32


def _select_head_layer_names(layer_names, prefixes=HEAD_NAME_PREFIXES):
    """Return the set of layer names that fall under the Detect head.

    Pure string-prefix match — no TRT / onnx / GPU dependency — so the head
    policy is unit-testable without the tensorrt wheels (the testable seam;
    mirrors ``src/quantize.py::_resolve_node_names_by_prefix`` as the testable
    half of the ORT path's head protection). The TRT-touching half
    (:func:`_exclude_head_layers`) consumes this on the parsed ``network``.
    """
    if not prefixes:
        return set()
    return {
        name for name in layer_names
        if name and any(name.startswith(p) for p in prefixes)
    }


def _is_qdq_onnx(onnx_path) -> bool:
    """True iff the ONNX at ``onnx_path`` carries explicit QDQ nodes.

    Pure (loads the graph via ``onnx`` with no external data; no TRT / GPU).
    The testable seam for the QDQ-vs-plain-ONNX branch in
    :func:`build_tensorrt_engine` — mirrors :func:`_select_head_layer_names` as
    the pure half of the policy.

    A QDQ model quantizes per its ``QuantizeLinear`` / ``DequantizeLinear``
    nodes (explicit quantization); the YOLOv8 Detect head (``/model.22/``),
    which carries none (excluded by :func:`src.quantize.quantize_onnx_to_int8`),
    stays FP32 by construction — no ``layer.precision`` /
    ``OBEY_PRECISION_CONSTRAINTS`` needed. Unlike the calibrator+OBEY path,
    this mechanism actually HOLDS on T4 / TRT 10.4, where calibrator+OBEY was
    proven not to (ablation wash: head-"excluded" == whole-net, both collapse
    ``detection_fail_rate`` 50–66%; the precision constraints don't bind under
    calibrator-based implicit quantization).
    """
    try:
        import onnx
        model = onnx.load_model(str(onnx_path), load_external_data=False)
    except Exception as e:  # pragma: no cover — best-effort
        logger.warning(
            "Could not load %s to check for QDQ nodes (%s); assuming plain "
            "(non-QDQ) ONNX — the calibrator+OBEY path will be used.",
            onnx_path, e,
        )
        return False
    return any(
        node.op_type in ("QuantizeLinear", "DequantizeLinear")
        for node in model.graph.node
    )


# ---------------------------------------------------------------------------
# INT8 Entropy calibrator — subclasses trt.IInt8EntropyCalibrator2 (with a
# duck-typed trt.IInt8Calibrator fallback for hosts where the v2 binding is
# missing). Rewired to preprocess via the SHARED preprocess_single so the
# calibration input distribution matches the deployed path exactly.
# ---------------------------------------------------------------------------
def _create_calibrator_class():
    if hasattr(trt, "IInt8EntropyCalibrator2"):
        base = trt.IInt8EntropyCalibrator2
    elif hasattr(trt, "IInt8Calibrator"):
        base = trt.IInt8Calibrator
    else:
        base = object

    class Int8EntropyCalibrator(base):
        """Streaming INT8 entropy calibrator (TRT EntropyCalibration2)."""

        def __init__(
            self,
            image_paths: List[str],
            batch_size: int,
            input_name: str,
            input_shape,            # (C, H, W)
            cache_path: Optional[str] = None,
        ):
            if base is not object:
                super().__init__()
            self.image_paths = list(image_paths)
            self.batch_size = batch_size
            self.input_name = input_name
            self.channels, self.height, self.width = input_shape
            self.cache_path = cache_path
            self.current_index = 0
            self.dtype = np.float32
            logger.info(
                "INT8 Calibrator: images=%d batch=%d",
                len(self.image_paths), self.batch_size,
            )

            # CUDA Driver init. Lean bindings return tuples; _cuda_call unpacks.
            # cuCtxCreate no longer exists — TensorRT has already created a
            # context by the time the calibrator runs; grab the current one.
            _cuda_call(cuda.cuInit(0))
            self._device = _cuda_call(cuda.cuDeviceGet(0))
            self._context = _cuda_call(cuda.cuCtxGetCurrent())
            if self._context is None:
                raise RuntimeError(
                    "No active CUDA context; TensorRT should have created one."
                )

            bytes_per_image = (
                self.channels * self.height * self.width
                * np.dtype(np.float32).itemsize
            )
            err, self.device_input = cuda.cuMemAlloc(
                bytes_per_image * self.batch_size
            )
            _cuda_check(err)
            logger.info(
                "Allocated %.2f MB for INT8 calibration.",
                bytes_per_image * self.batch_size / 1024 / 1024,
            )
            self.host_batch = np.empty(
                (self.batch_size, self.channels, self.height, self.width),
                dtype=np.float32,
            )

        # ---- preprocessing: REUSE the shared letterbox core ----
        def _preprocess_image(self, image_path: str) -> np.ndarray:
            """Letterbox + BGR->RGB + CHW + /255 via shared preprocess_single.

            Matches the deployed path (preprocess_imgs -> _assemble_batch does
            the same /255 after stacking) so the calibrator sees the exact
            input distribution the engine serves at inference time.
            """
            r = preprocess_single(image_path, imgsz=self.width)
            if r is None:
                raise RuntimeError(f"Unreadable calibration image: {image_path}")
            # preprocess_single returns CHW uint8; normalize like _assemble_batch.
            return r["img"].astype(np.float32) / 255.0

        def _load_batch(self) -> int:
            self.host_batch.fill(0.0)
            loaded = 0
            while (
                loaded < self.batch_size
                and self.current_index < len(self.image_paths)
            ):
                path = self.image_paths[self.current_index]
                try:
                    self.host_batch[loaded] = self._preprocess_image(path)
                    loaded += 1
                except Exception as e:
                    logger.warning("Skipping calibration image %s: %s", path, e)
                self.current_index += 1
            return loaded

        # ---- TRT calibrator interface ----
        def get_batch_size(self) -> int:
            return self.batch_size

        def get_algorithm(self):
            if hasattr(trt, "CalibrationAlgoType"):
                return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
            return None

        def get_batch(self, names):
            loaded = self._load_batch()
            if loaded == 0:
                logger.info("INT8 calibration completed.")
                return None
            if loaded < self.batch_size:
                self.host_batch[loaded:self.batch_size].fill(0.0)
            host = np.ascontiguousarray(self.host_batch)
            # The TRT context lives on another thread; push/pop for the copy.
            _cuda_call(cuda.cuCtxPushCurrent(self._context))
            try:
                _cuda_call(cuda.cuMemcpyHtoD(
                    self.device_input, host.ctypes.data, host.nbytes
                ))
            finally:
                _cuda_call(cuda.cuCtxPopCurrent())
            logger.info(
                "Calibration batch %d/%d",
                self.current_index, len(self.image_paths),
            )
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if self.cache_path is None:
                return None
            cache = Path(self.cache_path)
            if not cache.exists():
                return None
            logger.info("Loading calibration cache: %s", cache)
            return cache.read_bytes()

        def write_calibration_cache(self, cache):
            if self.cache_path is None:
                return
            cache_file = Path(self.cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(cache)
            logger.info("Calibration cache saved: %s", cache_file)

        def release(self) -> None:
            """Free the calibrator's GPU allocation explicitly (idempotent)."""
            if getattr(self, "_released", False):
                return
            try:
                if getattr(self, "device_input", None) is not None:
                    _cuda_call(cuda.cuMemFree(self.device_input))
            except Exception:
                pass
            finally:
                self.device_input = None
                self._released = True

        def __del__(self):  # best-effort safety net if release() wasn't called
            try:
                self.release()
            except Exception:
                pass

    return Int8EntropyCalibrator


# ---------------------------------------------------------------------------
# Path A: ONNX -> TensorRT engine (Python API)
# ---------------------------------------------------------------------------
def build_tensorrt_engine(
    onnx_path: str,
    output_path: str,
    *,
    imgsz: int = 640,
    fp16: bool = False,
    int8: bool = False,
    max_batch: int = 8,
    workspace_bytes: int = WORKSPACE_BYTES,
    calib_images: Optional[List[str]] = None,
    calib_cache_path: Optional[str] = None,
    device: int = 0,
    static: bool = False,
    exclude_head: bool = True,
) -> str:
    """Build a TensorRT engine from an ONNX model (Path A, Python API).

    The primary build path — full control over the optimization profile,
    workspace, and INT8 calibrator. ``calib_images`` is the calibration image
    list (the CLI builds it from the shared ``CalibrationSampler``); INT8
    requires it.

    ``static=True`` builds a batch=1 engine from a static-batch ONNX (no
    optimization profile). Use it on Turing (sm_75, e.g. T4), where the
    dynamic-batch DFL reshape forces a ``Shape``+``Slice`` subgraph TRT 10.4
    can't lower (``nbDims > Dims::MAX_DIMS``). On Ampere+ (sm_80+) leave
    ``static=False`` for a dynamic-batch (1..max_batch) engine.

    ``exclude_head`` (default True, INT8-only) pins the ``/model.22/`` Detect
    head to FP32 via ``layer.precision`` + ``set_output_type`` +
    ``OBEY_PRECISION_CONSTRAINTS`` — the TRT mirror of ORT invariant #3. Whole-
    net INT8 collapses the head's unbounded box-decode accumulation (T4 proof:
    ``detection_fail_rate=50%``); pinning the head to FP32 restores detection
    consistency (expected ``detection_fail_rate→0%``). No-op for FP16/FP32
    (whole-net FP16 is benign — FP16 has the exponent range INT8 lacks).
    """
    if not tensorrt_available():
        raise ImportError(
            "tensorrt is not installed. Run: "
            "pip install -r requirements-tensorrt.txt (on top of "
            "requirements-kaggle.txt)."
        )
    if fp16 and int8:
        raise ValueError("Cannot enable FP16 and INT8 simultaneously.")

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    # Detect explicit-quantization (QDQ) input. A QDQ ONNX carries its own
    # QuantizeLinear/DequantizeLinear nodes, so TRT quantizes per the graph and
    # the head (/model.22/, excluded by src.quantize.quantize_onnx_to_int8) stays
    # FP32 by construction — no calibrator, no layer.precision/OBEY. A plain
    # ONNX needs the calibrator (the legacy path; OBEY does NOT hold on T4).
    is_qdq = _is_qdq_onnx(onnx_path)
    if int8 and not calib_images and not is_qdq:
        raise ValueError(
            "INT8 calibrator build requires calibration images. The default "
            "INT8 path builds a QDQ ONNX (via src.quantize.quantize_onnx_to_int8 "
            "on the TRT-friendly ONNX) and needs no calibrator — pass that QDQ "
            "ONNX, or use the CLI default `tensorrt build --precision int8 "
            "--model X.pt` which produces it."
        )

    set_cuda_visible_devices(int(device))
    _cuda_call(cuda.cuInit(0))

    precision = "INT8" if int8 else "FP16" if fp16 else "FP32"
    trt_major = int(trt.__version__.split(".")[0]) if hasattr(trt, "__version__") else 10
    logger.info(
        "Building engine: %s | TRT %s | precision=%s | imgsz=%d | max_batch=%d",
        output_path, trt_major, precision, imgsz, max_batch,
    )

    # Structural ONNX gate — cheap, pure (no CUDA/ORT), clearer error than
    # TRT's OnnxParser. Does NOT run shape_inference or rewrite the on-disk
    # graph (that's onnxsim's job, deliberately disabled for TRT — see the
    # "TRT-friendly ONNX" note in docs/TENSORRT.md). TRT re-infers on parse;
    # the only authoritative TRT validation is the build itself (OnnxParser
    # fails fast on a malformed graph; the tactic-selection phase after it is
    # where TRT-specific incompatibilities like nbDims>MAX_DIMS surface).
    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(onnx_path)))
    except Exception as e:
        raise RuntimeError(
            f"ONNX structural check failed for {onnx_path}: {e}. "
            f"Re-export (python main.py export --no-simplify --opset 13) "
            f"or pass --model X.pt to auto-export a TRT-friendly ONNX."
        ) from e

    t0 = time.perf_counter()
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()

    # Explicit-batch network + ONNX parse.
    network_flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        try:
            network_flags |= int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        except TypeError:
            network_flags |= int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH.value)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(parser.get_error(i))
            raise RuntimeError("Failed parsing ONNX.")

    _configure_precision(builder, config, fp16, int8)

    if int8 and is_qdq:
        # Explicit-quantization (QDQ) path — the robust INT8 mechanism. TRT
        # quantizes per the graph's QuantizeLinear/DequantizeLinear nodes; the
        # head (/model.22/, excluded by src.quantize.quantize_onnx_to_int8) has
        # no QDQ nodes and stays FP32 by construction. No calibrator, no
        # layer.precision/OBEY — the latter was proven NOT to hold the head FP32
        # under calibrator-based implicit quantization on T4/TRT 10.4 (ablation
        # wash: head-"excluded" == whole-net, both collapse detection_fail_rate
        # 50–66%). This reuses ORT's exact head-exclusion scope (invariant #3).
        logger.info(
            "QDQ explicit quantization — head FP32 via ORT-excluded QDQ nodes "
            "(no calibrator / no OBEY_PRECISION_CONSTRAINTS; reuses ORT "
            "invariant #3's /model.22/ scope)."
        )
    elif int8 and exclude_head:
        # Legacy calibrator+OBEY path (--calibrator ablation). WARNING: this
        # does NOT hold the head FP32 on T4 / TRT 10.4 — calibrator-based
        # implicit INT8 ignores the precision constraints (proven: pinning 70
        # head layers + OBEY collapses identically to --no-exclude-head). Kept
        # as an ablation for the interview narrative; produces collapsing
        # engines. The default QDQ path above is the robust fix.
        logger.warning(
            "Calibrator+OBEY INT8 path (--calibrator ablation): on T4 / TRT "
            "10.4 this does NOT hold the /model.22/ head FP32 — the engine "
            "collapses (detection_fail_rate~50-66%%). Use the default QDQ path "
            "(drop --calibrator) for a consistent INT8 engine."
        )
        constraint_flag = None
        if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            constraint_flag = "OBEY_PRECISION_CONSTRAINTS"
        elif hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
            # Older TRT spelling fallback — warn-and-continue vs hard obey.
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            constraint_flag = "PREFER_PRECISION_CONSTRAINTS"
        if constraint_flag is None:
            logger.warning(
                "Head-exclusion: no precision-constraint BuilderFlag found in "
                "this TRT build — layer.precision pins will NOT be enforced; "
                "the head will likely run INT8 and collapse (upgrade TRT)."
            )
            constraint_flag = "<none>"
        n_pinned = _exclude_head_layers(network)
        logger.info(
            "Head-exclusion (legacy calibrator path): pinned %d layer(s) under "
            "%s to FP32 (constraint=%s — mirrors ORT invariant #3). WARNING: "
            "this proves constraints were SET, not that the builder HONORED "
            "them; on T4/TRT 10.4 it does not (use the QDQ path).",
            n_pinned, HEAD_NAME_PREFIXES, constraint_flag,
        )

    calibrator = None
    if int8 and not is_qdq:
        # Calibrator only for the legacy plain-ONNX path. The QDQ path is
        # explicit-quantization — no calibrator (TRT reads scales from the QDQ
        # graph). A static-batch=1 network calibrates at batch=1 (its input is
        # fixed [1,3,H,W]); a dynamic network calibrates at up to 8.
        calib_bs = 1 if static else min(8, len(calib_images))
        calibrator = _create_calibrator_class()(
            image_paths=calib_images,
            batch_size=calib_bs,
            input_name="images",
            input_shape=(3, imgsz, imgsz),
            cache_path=calib_cache_path,
        )
        _set_calibrator(config, calibrator)

    # Optimization profile — dynamic batch 1..max_batch. Skipped for a
    # static-batch ONNX: a static input dim accepts no profile, and the engine
    # is batch=1. Static is the Turing (sm_75) route — the dynamic DFL reshape
    # over a symbolic batch forces a Shape+Slice subgraph TRT 10.4 can't lower
    # on Turing (nbDims > Dims::MAX_DIMS); a static-batch ONNX has no symbolic
    # dim, so no Shape subgraph, so no failure. On Ampere+ use dynamic.
    if not static:
        profile = builder.create_optimization_profile()
        min_shape = (1, 3, imgsz, imgsz)
        opt_shape = (max(1, max_batch // 2), 3, imgsz, imgsz)
        max_shape = (max_batch, 3, imgsz, imgsz)
        profile.set_shape("images", min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)
        logger.info("Optimization profile: min=%s opt=%s max=%s",
                    min_shape, opt_shape, max_shape)
    else:
        logger.info("Static-batch=1 build (no optimization profile) — Turing route")

    # Workspace (TRT 10 MemoryPoolType; legacy set_max_workspace_size fallback).
    if hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:  # pragma: no cover — older TRT
        config.set_max_workspace_size(workspace_bytes)
    logger.info("Workspace: %.1f GB", workspace_bytes / 1024 ** 3)

    logger.info("Building TensorRT engine...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        # The signature of the Turing dynamic-batch failure: a TRT tactic error
        # referencing nbDims > Dims::MAX_DIMS / an ONNXTRT_ShapeSlice node.
        # Direct the user at --static (Turing route) or Path B (Ultralytics).
        head_hint = ""
        if int8 and not is_qdq and exclude_head:
            # Calibrator-path only. If the build died under
            # OBEY_PRECISION_CONSTRAINTS (no FP32 impl for a head op — unlikely
            # for YOLOv8's Conv/ElementWise/Concat/Shuffle/Slice/Activation),
            # relaxing to PREFER lets the builder warn-and-continue. The conf-
            # cliff gate is the ground-truth check either way: if
            # detection_fail_rate=0 after a PREFER build, the head stayed
            # high-precision enough; if >0, the constraint didn't hold (the
            # documented T4 outcome — switch to the QDQ path, drop --calibrator).
            head_hint = (
                " If the TRT log reports a precision-constraint violation "
                "(no FP32 implementation for a /model.22/ head layer), "
                "relax OBEY_PRECISION_CONSTRAINTS by passing --no-exclude-head "
                "(ablation) or fall back to Path B."
            )
        raise RuntimeError(
            "TensorRT failed to build engine. If the TRT log shows "
            "'nbDims ... greater than Dims::MAX_DIMS' / "
            "'ONNXTRT_ShapeSlice' (the dynamic-batch DFL Shape subgraph, "
            "which TRT 10.4 cannot lower on Turing/sm_75 e.g. T4), rebuild "
            "with --static (static-batch=1 ONNX, no symbolic dim -> no Shape "
            "subgraph): `python main.py tensorrt build --model X.pt "
            "--static --precision fp16`. On Ampere+ (sm_80+) the default "
            "dynamic build works. Alternatively use Path B: "
            "`python main.py tensorrt export --model X.pt --precision fp16`."
            + head_hint
        )
    if calibrator is not None:
        calibrator.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)
    logger.info(
        "Engine saved: %s (%.1f MB, %.0f ms)",
        output_path, output_path.stat().st_size / 1024 / 1024,
        (time.perf_counter() - t0) * 1000.0,
    )
    return str(output_path)


def _exclude_head_layers(network, prefixes=HEAD_NAME_PREFIXES) -> int:
    """Pin every ``/model.22/`` Detect-head **float-compute** layer to FP32 (TRT 10).

    The TRT Path-A equivalent of ORT invariant #3's whole-head FP32 skip. After
    ``OnnxParser.parse``, ``layer.name`` carries the original ONNX node name, so
    a substring/prefix match over ``/model.22/`` selects the whole head subgraph
    (Conv / ElementWise / Concat / Shuffle / Slice / Activation — multiple TRT
    layers share the prefix; all eligible are tagged). Each is pinned via
    ``layer.precision = trt.float32`` + ``set_output_type``; the caller's
    ``OBEY_PRECISION_CONSTRAINTS`` makes the builder honor it instead of silently
    relaxing back to INT8 for speed.

    Must run on the parsed ``INetworkDefinition`` BEFORE
    ``build_serialized_network`` — fusion renames layers at build time, so the
    ``/model.22/`` prefix only matches pre-build.

    **Non-float layers are skipped**, not pinned: integer-output layers under
    ``/model.22/`` (``Constant`` holding Int64 shape tensors, ``Shape`` layers)
    are structural metadata, not quantized compute — and forcing Float precision
    on them fails the build ("cannot use precision Float with weights of type
    Int64"). Shape tensors are never quantized anyway, so skipping is safe.
    Returns the count pinned.
    """
    float_dtypes = {trt.float32, trt.float16}
    all_names = [network.get_layer(i).name for i in range(network.num_layers)]
    head_names = _select_head_layer_names(all_names, prefixes)
    pinned = 0
    skipped = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.name not in head_names:
            continue
        try:
            out_types = {layer.get_output_type(o) for o in range(layer.num_outputs)}
        except Exception:  # pragma: no cover — defensive; skip what can't be read
            skipped += 1
            continue
        # Only pin layers whose outputs are all float-valued. Integer/Bool outputs
        # (Constant Int64 shape tensors, Shape layers) reject a Float precision
        # constraint at build time and aren't quantized compute anyway.
        if not out_types or not out_types.issubset(float_dtypes):
            skipped += 1
            continue
        layer.precision = trt.float32
        for o in range(layer.num_outputs):
            layer.set_output_type(o, trt.float32)
        pinned += 1
    if skipped:
        logger.info(
            "Head-exclusion: skipped %d non-float /model.22/ layer(s) "
            "(Constant/Shape int tensors — not pin-able, not quantized).",
            skipped,
        )
    return pinned


def _configure_precision(builder, config, fp16: bool, int8: bool) -> None:
    """Set TF32 / FP16 / INT8 builder flags (platform-capability guarded)."""
    if hasattr(trt.BuilderFlag, "TF32"):
        if getattr(builder, "platform_has_tf32", False):
            config.set_flag(trt.BuilderFlag.TF32)
            logger.info("TF32 enabled")
        else:
            logger.info("TF32 unsupported on this GPU.")
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 enabled")
        else:
            raise RuntimeError("GPU does not support FP16.")
    if int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("INT8 enabled")
        else:
            raise RuntimeError("GPU does not support INT8.")


def _set_calibrator(config, calibrator) -> None:
    """Attach the INT8 calibrator to the builder config (API-version tolerant)."""
    if hasattr(config, "set_int8_calibrator"):
        config.set_int8_calibrator(calibrator)
    elif hasattr(config, "int8_calibrator"):
        config.int8_calibrator = calibrator
    else:
        raise RuntimeError(
            "TensorRT does not expose an INT8 calibrator interface."
        )


# ---------------------------------------------------------------------------
# Path B: .pt -> .engine via Ultralytics native export
# ---------------------------------------------------------------------------
def export_tensorrt_ultralytics(
    pt_path: str,
    engine_path: str,
    *,
    imgsz: int = 640,
    precision: str = "fp16",
    data_yaml: Optional[str] = None,
    device: int = 0,
) -> str:
    """Export a YOLOv8 ``.pt`` directly to a TensorRT ``.engine`` via
    Ultralytics' ``model.export(format="engine")`` (Path B).

    Kaggle/Colab route — no ``trtexec`` needed. ``precision`` selects FP32 /
    FP16 / INT8; the two are independent flags (``half`` controls FP16 only,
    ``int8`` selects INT8 calibration). INT8 requires ``data_yaml`` for
    calibration images.
    """
    if not tensorrt_available():
        raise ImportError(
            "tensorrt is not installed. Run: "
            "pip install -r requirements-tensorrt.txt (on top of "
            "requirements-kaggle.txt)."
        )
    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError(f"Unsupported precision: {precision!r}. Use fp32/fp16/int8.")
    is_int8 = precision == "int8"
    is_fp16 = precision == "fp16"
    if is_int8 and not data_yaml:
        raise ValueError("INT8 export requires data_yaml for calibration images.")

    from ultralytics import YOLO  # lazy — heavy dep, only on the Path-B build

    pt_path = Path(pt_path)
    engine_path = Path(engine_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"Model not found: {pt_path}")

    logger.info("Exporting %s -> %s via Ultralytics (precision=%s)",
                pt_path.name, engine_path, precision)
    model = YOLO(str(pt_path))
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    kwargs = {
        "format": "engine",
        "half": is_fp16,
        "int8": is_int8,
        "imgsz": imgsz,
        "verbose": False,
        "device": device,
    }
    if is_int8 and data_yaml is not None:
        kwargs["data"] = data_yaml
    model.export(**kwargs)

    exported = _find_exported_engine(pt_path, engine_path)
    if exported is None:
        raise RuntimeError("Ultralytics export did not produce an engine file.")
    if str(exported) != str(engine_path):
        import shutil
        shutil.move(str(exported), str(engine_path))
        logger.info("Moved engine to: %s", engine_path)

    logger.info(
        "Ultralytics TRT export complete -> %s (%.1f MB, %.0f ms)",
        engine_path, engine_path.stat().st_size / 1024 / 1024,
        (time.perf_counter() - t0) * 1000.0,
    )
    return str(engine_path)


def _find_exported_engine(pt_path: Path, engine_path: Path) -> Optional[Path]:
    """Locate the ``.engine`` Ultralytics emitted next to the ``.pt``."""
    model_dir = pt_path.parent
    stem = pt_path.stem
    candidate = model_dir / f"{stem}_engine" / f"{stem}.engine"
    if candidate.exists():
        return candidate
    for p in sorted(model_dir.rglob("*.engine")):
        return p
    if engine_path.exists():
        return engine_path
    return None
