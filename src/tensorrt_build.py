"""TensorRT engine build + Ultralytics native export for YOLOv8s.

Two build paths live here; ``trtexec`` is the third route:

* **Path A — Python API** (:func:`build_tensorrt_engine`): ``.onnx -> .engine``
  via ``trt.Builder`` + ``OnnxParser`` + ``OptimizationProfile`` +
  ``BuilderFlag.FP16``/``INT8`` + ``IInt8EntropyCalibrator2``. The primary
  build path: full control over profiles/workspace/precision and the INT8
  calibrator.

* **Path B — Ultralytics native** (:func:`export_tensorrt_ultralytics`):
  ``.pt -> .engine`` via ``model.export(format="engine")`` — Ultralytics
  invokes the TRT Python API internally in one step, with its own
  profile/workspace/precision defaults. Recommended for Kaggle / Colab where
  ``trtexec`` is unavailable.

Both paths produce the same raw ``[B, 4+nc, 8400]`` head that
:class:`src.tensorrt_engine.TensorRTEngine` consumes. Path A's calibration
image list is built by the shared :class:`src.sampler.CalibrationSampler`;
Path B hands ``data=data_yaml`` to Ultralytics, which draws its own
calibration set. The per-image preprocessing depends on the INT8 branch —
the default explicit-QDQ path quantizes the ONNX first and its
calibrator-free reader reuses :func:`src.preprocess.preprocess_imgs` (via
:class:`src.quantize.YOLOv8CalibrationDataReader`), while the
``--calibrator`` implicit-quantization path reuses
:func:`src.preprocess.preprocess_single` (letterbox + BGR->RGB + CHW; the
calibrator applies /255). Either way the calibration inputs go through the
same preprocessing core as the deployed path.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from utils import get_logger
from .preprocess import preprocess_single
from .quantize import HEAD_NAME_PREFIXES  # shared head-skip scope (see below)
# Reuse the engine module's guarded ``trt`` / ``cuda`` imports + the lean-bindings
# helpers — one CUDA dialect end-to-end, one availability predicate.
from .tensorrt_engine import (
    _cuda_call,
    _cuda_check,
    cuda,
    restore_cuda_visible_devices,
    set_cuda_visible_devices,
    tensorrt_available,
    trt,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Build constants — the TRT-build workspace pool size.
# ---------------------------------------------------------------------------
WORKSPACE_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB — comfortable on a T4 (16 GiB)


def trt_build_available() -> bool:
    """True iff the build path can run (tensorrt + cuda-python installed)."""
    return tensorrt_available()


# ---------------------------------------------------------------------------
# Detect-head exclusion — pins the whole YOLOv8 Detect head (``/model.22/``
# ONNX name prefix: cv2 box convs, cv3 cls convs, DFL, Sigmoid, Concat,
# box-decode arithmetic) to FP32 under an INT8 build. The head's box-decode
# accumulation is unbounded (no exponent range), so quantizing that subgraph
# to INT8 diverges box coordinates and flips low-confidence detections; FP16
# retains an exponent range, so head FP32 protection applies to INT8 builds
# only. HEAD_NAME_PREFIXES is imported from ``src.quantize`` at the top of
# this module — the single source of truth for the Detect-head skip boundary:
# the ORT INT8 ``nodes_to_exclude`` scope and the TRT calibrator path's head
# pinning both consume this import.


def _select_head_layer_names(layer_names, prefixes=HEAD_NAME_PREFIXES):
    """Return the set of layer names that fall under the Detect head.

    Pure string-prefix match — no TRT / onnx / GPU dependency — so the head
    policy is unit-testable without the tensorrt wheels. The TRT-touching half
    (:func:`_exclude_head_layers`) consumes this on the parsed ``network``.
    """
    if not prefixes:
        return set()
    return {
        name for name in layer_names
        if name and any(name.startswith(p) for p in prefixes)
    }


def _input_batch_build_error(batch_dim: int, static: bool) -> Optional[str]:
    """Mismatch message for the ONNX input batch dim vs ``--static``, or
    ``None`` when the combination is buildable.

    Pure (int in, str out) — no TRT dependency — so the build-gate policy is
    unit-testable without the tensorrt wheels. Gating here keeps the error at
    argument level; when the same mismatch reaches the TRT layer it reports
    as a profile-validation / engine-load error.
    """
    if static and batch_dim != 1:
        return (
            "--static builds a batch=1 engine, but the ONNX input batch is "
            f"{'dynamic' if batch_dim < 0 else batch_dim}. Re-export a "
            "static batch=1 ONNX (--no-dynamic) or pass a YOLOv8 .pt, which "
            "auto-exports it under --static."
        )
    if not static and batch_dim != -1:
        if batch_dim == 1:
            return (
                "The ONNX input batch is the concrete dim 1, so the dynamic "
                "optimization profile (1..max_batch) cannot be set on it. "
                "Pass --static to build the batch=1 static engine this ONNX "
                "supports, or pass a YOLOv8 .pt without --static, which "
                "auto-exports a dynamic-batch TRT-friendly ONNX."
            )
        return (
            f"The ONNX input batch is the concrete dim {batch_dim}; this "
            "build path supports a dynamic batch (1..max_batch, no --static) "
            "or a static batch=1 engine (--static). Pass a YOLOv8 .pt "
            "(auto-exports a dynamic-batch TRT-friendly ONNX) or re-export "
            "this ONNX with a dynamic batch."
        )
    return None


def _is_qdq_onnx(onnx_path) -> bool:
    """True iff the ONNX at ``onnx_path`` carries explicit QDQ nodes.

    Pure (loads the graph via ``onnx`` with no external data; no TRT / GPU);
    the seam for the QDQ-vs-plain-ONNX branch in
    :func:`build_tensorrt_engine`.

    A QDQ model quantizes per its ``QuantizeLinear`` / ``DequantizeLinear``
    nodes (explicit quantization); the YOLOv8 Detect head (``/model.22/``),
    which carries none (excluded by :func:`src.quantize.quantize_onnx_to_int8`),
    stays FP32 by construction — no ``layer.precision`` /
    ``OBEY_PRECISION_CONSTRAINTS`` needed.
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
# missing). Calibration inputs go through the shared preprocess_single, so
# the calibration input distribution matches the deployed path.
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
            self.fed_batches = 0
            self.dtype = np.float32
            logger.info(
                "INT8 Calibrator: images=%d batch=%d",
                len(self.image_paths), self.batch_size,
            )

            # CUDA Driver init. Lean bindings return tuples; _cuda_call unpacks.
            # The calibrator runs on the build thread, which
            # ``build_tensorrt_engine`` has made current (primary context):
            # adopt it for the input allocation and the H2D copies.
            _cuda_call(cuda.cuInit(0))
            self._device = _cuda_call(cuda.cuDeviceGet(0))
            self._context = _cuda_call(cuda.cuCtxGetCurrent())
            if self._context is None:
                raise RuntimeError(
                    "No active CUDA context on the build thread. TensorRT "
                    "does not create CUDA contexts — the caller must push "
                    "one (build_tensorrt_engine does)."
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
            """Letterbox + BGR->RGB + CHW via shared preprocess_single;
            /255 applied here.

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
                if self.fed_batches == 0:
                    logger.warning(
                        "INT8 calibration has no loadable images "
                        "(%d candidate paths, all skipped).",
                        self.current_index,
                    )
                else:
                    logger.info(
                        "INT8 calibration completed (%d batches).",
                        self.fed_batches,
                    )
                return None
            self.fed_batches += 1
            if loaded < self.batch_size:
                # The trailing partial batch is zero-filled (all-black
                # images) before the H2D copy — TRT's calibrator API takes
                # fixed-size batches. The QDQ path's
                # YOLOv8CalibrationDataReader yields a true partial batch;
                # the two validated calibration paths differ in this
                # tail-batch detail.
                self.host_batch[loaded:self.batch_size].fill(0.0)
            host = np.ascontiguousarray(self.host_batch)
            # get_batch runs synchronously on the build thread inside
            # build_serialized_network, where build_tensorrt_engine keeps the
            # primary context current — the H2D copy needs no extra push.
            _cuda_call(cuda.cuMemcpyHtoD(
                self.device_input, host.ctypes.data, host.nbytes
            ))
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
            """Free the calibrator's GPU allocation explicitly (idempotent).

            The allocation lives under the build thread's primary context, so
            the free runs with that context pushed — the handle was captured
            at init (``cuCtxGetCurrent``) while ``build_tensorrt_engine`` held
            the context current for the whole build. ``_build_engine`` calls
            ``release()`` on both the success and the build-failure path,
            while the context is still current; ``__del__`` remains a
            best-effort net for paths that bypass it.
            """
            if getattr(self, "_released", False):
                return
            try:
                if getattr(self, "device_input", None) is not None:
                    _cuda_call(cuda.cuCtxPushCurrent(self._context))
                    try:
                        _cuda_call(cuda.cuMemFree(self.device_input))
                    finally:
                        _cuda_call(cuda.cuCtxPopCurrent())
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
    list (callers build it from the shared ``CalibrationSampler``); the
    calibrator path requires it — the default QDQ path needs none (the ONNX
    is quantized before the build).

    ``static=True`` builds a batch=1 engine from a static-batch ONNX (no
    optimization profile). Use it on Turing (sm_75, e.g. T4): the validated
    T4 session hit a tactic-selection failure on a dynamic-batch build
    around the DFL ``Shape``+``Slice`` subgraph (an observed signature
    quoting ``nbDims > Dims::MAX_DIMS``; the cause is inferred, not
    documented by NVIDIA). On
    Ampere+ (sm_80+) ``static=False`` (a dynamic-batch (1..max_batch)
    engine) is the expected default.

    ``exclude_head`` (default True, INT8-only) pins the ``/model.22/`` Detect
    head to FP32 via ``layer.precision`` + ``set_output_type`` +
    ``OBEY_PRECISION_CONSTRAINTS`` (or the older
    ``PREFER_PRECISION_CONSTRAINTS`` spelling on builds without it), keeping
    the head's box-decode in FP32 under an INT8 build. No-op for FP16/FP32
    (whole-net FP16 is benign — FP16 has the exponent range INT8 lacks) and
    for a QDQ input (the head's FP32 there comes from its missing QDQ
    nodes).
    """
    if not tensorrt_available():
        raise ImportError(
            "tensorrt is not installed. Run: "
            "pip install -r requirements-tensorrt.txt (on top of "
            "requirements-kaggle.txt)."
        )
    if fp16 and int8:
        raise ValueError("Cannot enable FP16 and INT8 simultaneously.")
    if int(max_batch) < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    # Detect explicit-quantization (QDQ) input. A QDQ ONNX carries its own
    # QuantizeLinear/DequantizeLinear nodes, so TRT quantizes per the graph and
    # the head (/model.22/, excluded by src.quantize.quantize_onnx_to_int8) stays
    # FP32 by construction — no calibrator, no layer.precision/OBEY. A plain
    # ONNX requires calibration images for the calibrator path.
    is_qdq = _is_qdq_onnx(onnx_path)
    if int8 and not calib_images and not is_qdq:
        raise ValueError(
            "INT8 calibrator build requires calibration images. The default "
            "INT8 path builds a QDQ ONNX (via src.quantize.quantize_onnx_to_int8 "
            "on the TRT-friendly ONNX) and needs no calibrator — pass that "
            "QDQ ONNX."
        )
    if int8 and is_qdq and calib_images:
        # A QDQ ONNX is consumed through the graph's own scales — the
        # calibrator is not created (see _build_engine), so a supplied
        # calibration list would do nothing.
        logger.warning(
            "%s already carries QDQ nodes; the calibrator path is skipped "
            "and the calibration images are unused.",
            onnx_path.name,
        )

    prev_cvd = set_cuda_visible_devices(int(device))
    try:
        _cuda_call(cuda.cuInit(0))
        # After cuInit under the pin, CUDA re-enumerates the filtered set: the
        # requested physical GPU is ordinal 0 (cuDeviceGet(device) would
        # address the (device+1)-th visible device and raise
        # CUDA_ERROR_INVALID_DEVICE on the last GPU) — the same convention
        # the engine follows.
        #
        # Precondition (see set_cuda_visible_devices' docstring): the env pin
        # only takes effect if no CUDA context has been initialized in this
        # process yet — including an ORT CUDA-EP session created on another
        # GPU id. The CLI's default INT8 flow quantizes the ONNX via
        # quantize_onnx_to_int8 (a CUDA-EP ORT session) BEFORE reaching this
        # line, which initializes CUDA driver-wide; the re-enumeration above
        # then does not apply. On the validated single-GPU T4 setup this is
        # unobservable; on multi-GPU hosts, run each build in a fresh process
        # for the pin to be honored.
        device_handle = _cuda_call(cuda.cuDeviceGet(0))
    finally:
        # Same restore contract as the engine's __init__: the pin exists for
        # this enumeration only, and every exit path — including a failed
        # cuInit or cuDeviceGet — returns the caller's env value.
        restore_cuda_visible_devices(prev_cvd)
    ctx = _cuda_call(cuda.cuDevicePrimaryCtxRetain(device_handle))
    # Builder creation, ONNX parse, tactic selection, and the calibrator's
    # allocations all run on this thread: make the primary context current
    # for the whole build. The calibrator adopts the same context through
    # cuCtxGetCurrent.
    pushed = False
    try:
        _cuda_call(cuda.cuCtxPushCurrent(ctx))
        pushed = True
        return _build_engine(
            onnx_path=onnx_path,
            output_path=output_path,
            imgsz=imgsz,
            fp16=fp16,
            int8=int8,
            is_qdq=is_qdq,
            max_batch=max_batch,
            workspace_bytes=workspace_bytes,
            calib_images=calib_images,
            calib_cache_path=calib_cache_path,
            static=static,
            exclude_head=exclude_head,
        )
    finally:
        if pushed:
            # Pop only the context pushed above; a failed push leaves the
            # calling thread's stack as it was (same guard as the engine).
            try:
                _cuda_call(cuda.cuCtxPopCurrent())
            except Exception:
                pass
        try:
            # Exception-guarded like the pop above: a failed release must
            # not mask the build's own result or exception. The release
            # also covers a failed push — the retention happened above it.
            _cuda_call(cuda.cuDevicePrimaryCtxRelease(device_handle))
        except Exception:
            pass


def _build_engine(
    onnx_path: Path,
    output_path: Path,
    *,
    imgsz: int,
    fp16: bool,
    int8: bool,
    is_qdq: bool,
    max_batch: int,
    workspace_bytes: int,
    calib_images: Optional[List[str]],
    calib_cache_path: Optional[str],
    static: bool,
    exclude_head: bool,
) -> str:
    """Build the serialized engine; runs with the primary context current.

    Called from :func:`build_tensorrt_engine`, which owns the CUDA context
    push/pop and the primary-context release.
    """
    precision = "INT8" if int8 else "FP16" if fp16 else "FP32"
    trt_major = int(trt.__version__.split(".")[0]) if hasattr(trt, "__version__") else 10
    logger.info(
        "Building engine: %s | TRT %s | precision=%s | imgsz=%d | max_batch=%d",
        output_path, trt_major, precision, imgsz, max_batch,
    )

    # Structural ONNX gate — cheap and pure (no CUDA/ORT); catches a
    # malformed graph before the parse. Does NOT run shape_inference or rewrite the on-disk
    # graph (onnxsim is disabled for TRT — a simplified dynamic Shape+Slice
    # subgraph is one trigger of the Turing dynamic-shape limit). TRT re-infers
    # on parse; the build itself is the final TRT validation (OnnxParser fails
    # fast on a malformed graph; the tactic-selection phase after it is where
    # TRT-specific incompatibilities like nbDims>MAX_DIMS surface).
    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(onnx_path)))
    except Exception as e:
        raise RuntimeError(
            f"ONNX structural check failed for {onnx_path}: {e}. "
            f"Re-export a TRT-friendly ONNX (opset 13, no simplify) or "
            f"pass a YOLOv8 .pt to auto-export one."
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

    if network.num_inputs != 1:
        raise RuntimeError(
            f"ONNX has {network.num_inputs} inputs; the build path supports "
            "exactly one image input."
        )
    input_name = network.get_input(0).name
    # Argument-level gate on the input batch dim vs --static (see
    # _input_batch_build_error for the policy and the messages).
    batch_error = _input_batch_build_error(
        int(network.get_input(0).shape[0]), static
    )
    if batch_error is not None:
        raise ValueError(batch_error)

    _configure_precision(builder, config, fp16, int8)

    if int8 and is_qdq:
        # Explicit-quantization (QDQ) path — TRT quantizes per the graph's
        # QuantizeLinear/DequantizeLinear nodes; the head (/model.22/, excluded
        # by src.quantize.quantize_onnx_to_int8) has no QDQ nodes and stays FP32
        # by construction. No calibrator, no layer.precision/OBEY.
        if not exclude_head:
            logger.warning(
                "--no-exclude-head has no effect on a QDQ ONNX: the head's "
                "FP32 protection comes from its absent QDQ nodes (this path "
                "sets no layer precision pins)."
            )
        logger.info(
            "QDQ explicit quantization — head FP32 via excluded QDQ nodes "
            "(no calibrator / no OBEY_PRECISION_CONSTRAINTS)."
        )
    elif int8 and exclude_head:
        # Calibrator path (--calibrator): attaches an INT8 calibrator and pins
        # head layers to FP32; constraint enforcement depends on the
        # BuilderFlag set below.
        logger.warning(
            "Calibrator+OBEY INT8 path (--calibrator): under "
            "calibrator-based implicit INT8 the precision constraints may "
            "not bind at tactic selection; the /model.22/ head may then run "
            "INT8. The default QDQ path (drop --calibrator) holds the head "
            "FP32 by QDQ omission."
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
                "the head will likely run INT8 (upgrade TRT)."
            )
            constraint_flag = "<none>"
        n_pinned = _exclude_head_layers(network)
        logger.info(
            "Head-exclusion (calibrator path): pinned %d layer(s) under "
            "%s to FP32 (constraint=%s). The constraints were SET; under "
            "calibrator-based implicit INT8 they may still not bind at tactic "
            "selection (use the QDQ path for that).",
            n_pinned, HEAD_NAME_PREFIXES, constraint_flag,
        )

    calibrator = None
    try:
        if int8 and not is_qdq:
            # Calibrator only for the plain (non-QDQ) ONNX path. The QDQ path
            # is explicit-quantization — no calibrator (TRT reads scales from
            # the QDQ graph). A static-batch=1 network calibrates at batch=1
            # (its input is fixed [1,3,H,W]); a dynamic network calibrates at
            # up to the profile's max_batch — a calibrator batch above that
            # bound is rejected by the build.
            calib_bs = 1 if static else min(max_batch, len(calib_images))
            calibrator = _create_calibrator_class()(
                image_paths=calib_images,
                batch_size=calib_bs,
                input_name=input_name,
                input_shape=(3, imgsz, imgsz),
                cache_path=calib_cache_path,
            )
            _set_calibrator(config, calibrator)

        # Optimization profile — dynamic batch 1..max_batch. Skipped for a
        # static-batch ONNX: a static input dim accepts no profile, and the
        # engine is batch=1. Static is the Turing (sm_75) route — on the
        # validated T4 the dynamic DFL reshape over a symbolic batch produces
        # the observed tactic-selection failure (see the build-failure hint
        # below); a static-batch ONNX has no symbolic dim, so that subgraph
        # does not appear. On Ampere+ use dynamic.
        if not static:
            profile = builder.create_optimization_profile()
            min_shape = (1, 3, imgsz, imgsz)
            opt_shape = (max(1, max_batch // 2), 3, imgsz, imgsz)
            max_shape = (max_batch, 3, imgsz, imgsz)
            profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
            config.add_optimization_profile(profile)
            logger.info("Optimization profile: min=%s opt=%s max=%s",
                        min_shape, opt_shape, max_shape)
        else:
            logger.info("Static-batch=1 build (no optimization profile) — Turing route")

        # Workspace pool size (TRT 10 memory-pool API; set_max_workspace_size
        # on pre-10 builds).
        if hasattr(trt, "MemoryPoolType"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        else:  # pragma: no cover — older TRT
            config.set_max_workspace_size(workspace_bytes)
        logger.info("Workspace: %.1f GB", workspace_bytes / 1024 ** 3)

        logger.info("Building TensorRT engine...")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            # Dynamic-batch failure hint. The quoted signature is the
            # observation from the validated T4 (Turing/sm_75) session with
            # TRT 10.x — a tactic-selection error naming an ONNXTRT_ShapeSlice
            # node and an nbDims > Dims::MAX_DIMS clause; it is recorded as an
            # observed signature, not a documented TRT guarantee, and the
            # underlying cause (the dynamic batch dim feeding the DFL
            # Shape+Slice subgraph) is inferred, not confirmed by NVIDIA
            # docs. Direct the user at --static (Turing route) or Path B
            # (Ultralytics).
            head_hint = ""
            if int8 and not is_qdq and exclude_head:
                # Calibrator-path hint: a precision-constraint build
                # failure can be relaxed via --no-exclude-head (ablation);
                # the default QDQ path does not take this branch.
                head_hint = (
                    " If the TRT log reports a precision-constraint violation "
                    "(no FP32 implementation for a /model.22/ head layer), "
                    "relax OBEY_PRECISION_CONSTRAINTS by passing "
                    "--no-exclude-head (ablation) or fall back to Path B."
                )
            raise RuntimeError(
                "TensorRT failed to build engine. If the TRT log shows the "
                "signature observed on the validated Turing/sm_75 T4 session "
                "(TRT 10.x, e.g. 'nbDims ... greater than Dims::MAX_DIMS' / "
                "'ONNXTRT_ShapeSlice' around the dynamic-batch DFL Shape "
                "subgraph), rebuild with --static (static-batch=1 ONNX; a "
                "concrete batch dim keeps the Shape+Slice subgraph out of "
                "the graph): pass a YOLOv8 .pt with --static --precision "
                "fp16. On Ampere+ (sm_80+) the dynamic "
                "route is the expected default. Alternatively use Path B: "
                "export via Ultralytics (`model.export(format=\"engine\")`, "
                "see export_tensorrt_ultralytics)." + head_hint
            )
    finally:
        # Free the calibrator's GPU allocation while the build context is
        # still current — build_tensorrt_engine pops and releases the primary
        # context in its own finally, after which the free cannot run. The
        # calibrator is created inside this block so an early failure (profile
        # setup, workspace sizing) also reaches this release.
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

    After ``OnnxParser.parse``, ``layer.name`` carries the original ONNX node
    name, so a substring/prefix match over ``/model.22/`` selects the whole
    head subgraph (Conv / ElementWise / Concat / Shuffle / Slice / Activation
    — multiple TRT layers share the prefix; all eligible are tagged). Each is
    pinned via ``layer.precision = trt.float32`` + ``set_output_type``; the
    caller's ``OBEY_PRECISION_CONSTRAINTS`` makes the builder honor the pin;
    without it the builder may relax the pin back to INT8 at tactic selection.

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
        # Only pin layers whose outputs are all float-valued. Integer/Bool
        # outputs (Constant Int64 shape tensors, Shape layers) reject a Float
        # precision constraint at build time and aren't quantized compute.
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
    """Set the FP16 / INT8 builder flags (platform-capability guarded).

    TF32 is not touched here — the builder's default TF32 state applies
    (moot on the pinned T4: sm_75 has no TF32 either way).
    """
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

    Kaggle/Colab route — no ``trtexec`` needed. The engine is static
    batch=1 (``batch=1, dynamic=False``). ``precision`` selects FP32 /
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
    # Static batch=1 — the profile-less engine shape the loader's static
    # path expects; pinned explicitly.
    kwargs = {
        "format": "engine",
        "half": is_fp16,
        "int8": is_int8,
        "imgsz": imgsz,
        "batch": 1,
        "dynamic": False,
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
    """Locate the ``.engine`` Ultralytics emitted for ``pt_path``.

    Ultralytics writes ``{stem}.engine`` next to the ``.pt``; some versions
    nest it under ``{stem}_engine/``. Both layouts are globbed (top level of
    the ``.pt``'s directory, stem-scoped so unrelated leftover engines are
    excluded) and the newest match wins — stem-prefixed precision variants
    (``{stem}_int8.engine``) also match the glob, and the freshly exported
    engine is the most recently written one. ``engine_path`` is only
    consulted when no such file exists.
    """
    model_dir = pt_path.parent
    stem = pt_path.stem
    candidates = list(model_dir.glob(f"{stem}*.engine"))
    nested = model_dir / f"{stem}_engine" / f"{stem}.engine"
    if nested.exists():
        candidates.append(nested)
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    if engine_path.exists():
        return engine_path
    return None
