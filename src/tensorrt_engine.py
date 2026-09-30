"""TensorRT inference backend for YOLOv8s.

Mirrors :class:`src.openvino_engine.OpenVINOEngine`'s API (``infer`` /
``infer_frames`` / ``_run_batch`` with an explicit ``nc=``) so the CLI, the
benchmark harness, and the server hot path plug in unchanged — reusing the
shared ``src.preprocess`` + ``src.postprocess`` core like every other Python
backend. TensorRT is GPU-only and an *optional* dep; the module imports cleanly
without the ``tensorrt`` / ``cuda-python`` wheels so the model-free test suite
stays green.

TensorRT 10.x notes (the modern API this engine speaks end-to-end):

* **Tensor-name API only.** ``set_tensor_address(name, ptr)`` +
  ``execute_async_v3(stream)``. The legacy index-based ``set_binding_address``
  / ``enqueueV2`` family was removed upstream — there is no fallback.
* **``cuda.bindings.driver`` "lean" bindings (CUDA 12.x).** Every Driver API
  call returns a ``(CUresult, *values)`` tuple, normalised through
  :func:`_cuda_call` / :func:`_cuda_check`. No PyCUDA — one CUDA dialect
  end-to-end, matching :mod:`src.tensorrt_build`.
* **Context-managed.``cuDevicePrimaryCtxRetain`` + per-thread
  ``cuCtxPushCurrent`` / ``cuCtxPopCurrent``; ``release()`` tears the primary
  context, the user-created stream, and the device buffers down in
  deterministic order so the engine destructor is a true no-op.
* **Dynamic batch.``set_input_shape`` per forward resolves the optimization
  profile the engine was built with (1..max_batch). Any count in that range is
  accepted directly — no zero-pad (unlike a static-batch OpenVINO IR, whose DFL
  reshape constant is baked to a fixed batch).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from utils import get_logger, load_images, save_annotated_image
from . import post_process, preprocess_imgs, preprocess_frames
# A serialized TRT engine carries no ultralytics "names" metadata, so class
# names come from data.yaml — exactly the OpenVINO IR situation. Reuse the
# OpenVINO engine's pure-Python reader (same YOLO names field, same list/dict
# normalization) rather than forking it.
from .openvino_engine import _class_names_from_data_yaml

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional-dep guard — tensorrt + cuda-python are NOT in the base pin set.
# The guard mirrors OpenVINO's: importing the module without the wheels
# installed sets _TRT_AVAILABLE=False; constructing TensorRTEngine then raises
# ImportError with a helpful message (never AttributeError). The model-free
# test suite asserts that contract.
# ---------------------------------------------------------------------------
try:  # pragma: no cover — exercised on GPU hosts
    import tensorrt as trt
    from cuda.bindings import driver as cuda  # lean bindings, CUDA 12.x
    _TRT_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised on CPU CI / test runners
    trt = None  # type: ignore[assignment]
    cuda = None  # type: ignore[assignment]
    _TRT_AVAILABLE = False


def tensorrt_available() -> bool:
    """True if the ``tensorrt`` + ``cuda-python`` wheels are importable."""
    return _TRT_AVAILABLE


# ---------------------------------------------------------------------------
# Layout constants — output buffer sizing when the engine reports a dynamic
# axis. Mirrors the standalone project's constants module (the only TRT-relevant
# pieces; the rest of that module is dropped — shared utils own image suffixes
# / palettes).
# ---------------------------------------------------------------------------
YOLOV8_BOX_OFFSET = 4  # [cx, cy, w, h] before the class scores
COCO_MAX_CLASSES = 80   # conservative upper bound for COCO-family models


# ---------------------------------------------------------------------------
# cuda.bindings.driver "lean" helpers — folded here (single source of truth
# for the tuple-return style used by the engine). The build module reuses the
# same idiom inline. Lean bindings return (CUresult, *values) for every call.
# ---------------------------------------------------------------------------
def _normalize_err(err) -> object:
    """Pull a bare CUresult out of either binding shape (fat enum / lean tuple)."""
    if isinstance(err, tuple):
        if not err:
            raise RuntimeError("CUDA Driver API returned an empty tuple")
        err = err[0]
    return err


def _cuda_call(result):
    """Validate a lean-bindings return tuple and unpack the payload.

    ``(CUresult,)`` -> ``None``; ``(CUresult, v)`` -> ``v``;
    ``(CUresult, a, b, ...)`` -> ``(a, b, ...)``. Raises on any non-success
    CUresult. Used at every ``cuda.<fn>(...)`` call site.
    """
    if not isinstance(result, tuple):
        raise RuntimeError(f"Unexpected CUDA return: {result!r}")
    err = _normalize_err(result[0])
    if cuda is not None and err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA Driver API failed: {err}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def _cuda_check(err) -> None:
    """Raise ``RuntimeError`` if a CUresult is non-success (bare or tuple form)."""
    err = _normalize_err(err)
    if cuda is not None and err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA Driver API failed: {err}")


def set_cuda_visible_devices(device_id: int) -> None:
    """Pin ``CUDA_VISIBLE_DEVICES`` to one device — BEFORE any CUDA init."""
    if device_id < 0:
        raise ValueError(f"device_id must be >= 0, got {device_id}")
    target = str(device_id)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != target:
        os.environ["CUDA_VISIBLE_DEVICES"] = target


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class TensorRTEngine:
    """TensorRT inference engine for YOLOv8.

    Parameters
    ----------
    model_path
        Path to a serialized TensorRT ``.engine`` / ``.plan``.
    device
        GPU id as an int, or ``"cuda"`` / ``"cuda:N"``. Default ``0``.
    imgsz
        Square network input size (default 640). Must match the engine.
    max_batch
        Upper bound of the optimization profile the engine was built with
        (default 8). A deserialized engine does not expose its profile bounds,
        so this is the caller's assertion; forwards are capped at it.
    class_names
        Optional ``{int: str}`` override; otherwise read from ``data_yaml``.
    data_yaml
        Path to the YOLO ``data.yaml`` carrying the class-name list — the
        canonical source, since a serialized engine has no metadata.
    """

    def __init__(
        self,
        model_path: str,
        device: Union[int, str] = 0,
        imgsz: int = 640,
        max_batch: int = 8,
        class_names: Optional[dict] = None,
        data_yaml: Optional[Union[str, Path]] = None,
    ) -> None:
        if not _TRT_AVAILABLE:
            raise ImportError(
                "tensorrt is not installed. Run: "
                "pip install -r requirements-tensorrt.txt (on top of "
                "requirements-kaggle.txt)."
            )

        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.max_batch = int(max_batch)
        self.class_names = class_names

        device_id = self._resolve_device_id(device)
        # CUDA_VISIBLE_DEVICES must be set before cuInit so the runtime only
        # sees the requested device.
        set_cuda_visible_devices(device_id)
        _cuda_call(cuda.cuInit(0))
        self.device_id = device_id
        self._device = _cuda_call(cuda.cuDeviceGet(device_id))
        # Lean bindings have no cuCtxCreate; adopt the driver/runtime primary
        # context so we can cuCtxPushCurrent / cuCtxPopCurrent per thread.
        self._cuda_ctx = _cuda_call(cuda.cuDevicePrimaryCtxRetain(self._device))

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Engine not found: {self.model_path}")

        self._load_engine()
        self._resolve_class_names(data_yaml)
        self._warmup()
        logger.info(
            "TensorRT engine initialized | %s | max_batch=%d | device=%d",
            Path(self.model_path).name, self.max_batch, device_id,
        )

    # ----------------------------------------------------------------- device
    @staticmethod
    def _resolve_device_id(device: Union[int, str]) -> int:
        """Coerce ``int | "cuda" | "cuda:N"`` to a numeric GPU id."""
        if isinstance(device, int):
            return device
        s = str(device).lower()
        if s == "cpu":
            raise ValueError("TensorRT is GPU-only; device='cpu' is invalid")
        if s in ("cuda", "cuda:0"):
            return 0
        if s.startswith("cuda:"):
            return int(s.split(":", 1)[1])
        # Already a numeric string.
        return int(s)

    # ------------------------------------------------------------- load engine
    def _load_engine(self) -> None:
        """Deserialize the engine and allocate the host/device buffers."""
        trt_logger = trt.Logger(trt.Logger.ERROR)
        with open(self.model_path, "rb") as f:
            engine_data = f.read()
        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()

        # Discover I/O tensor names (TRT 10 tensor-name API).
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        logger.info("TRT inputs: %s | outputs: %s",
                    self.input_names, self.output_names)

        input_shape = self.engine.get_tensor_shape(self.input_names[0])
        self.channels = int(input_shape[1]) if len(input_shape) > 1 else 3
        self.height = int(input_shape[2]) if len(input_shape) > 2 else 640
        self.width = int(input_shape[3]) if len(input_shape) > 3 else 640
        # An EfficientNMS-tailored export exposes a 4-D output; the raw YOLOv8
        # export (this project's path) exposes the 3-D [B, 4+nc, N] head.
        self._has_nms_plugin = len(
            self.engine.get_tensor_shape(self.output_names[0])
        ) == 4

        self._allocate_buffers()

    # -------------------------------------------------------- allocate buffers
    def _allocate_buffers(self) -> None:
        """Allocate host/device buffers + a CUDA stream, sized from max_batch.

        Output size is derived from the engine tensor shape with dynamic dims
        (-1) resolved to conservative maxima (max_batch / COCO_MAX_CLASSES /
        8400), so any in-profile batch fits the pre-allocated buffer.
        """
        input_size = (
            self.max_batch * self.channels * self.height * self.width
        )
        output_shape = self.engine.get_tensor_shape(self.output_names[0])
        concrete = list(output_shape)
        for i, d in enumerate(concrete):
            if d == -1:
                if i == 0:
                    concrete[i] = self.max_batch
                elif i == 1:
                    concrete[i] = YOLOV8_BOX_OFFSET + COCO_MAX_CLASSES
                else:
                    concrete[i] = 8400
        output_size = int(np.prod(concrete))
        logger.info("Output shape %s (concrete max %s)", output_shape, concrete)

        elem = np.dtype(np.float32).itemsize
        self._host_input = np.empty(input_size, dtype=np.float32)
        self._host_output = np.empty(output_size, dtype=np.float32)

        self._device_input = _cuda_call(
            cuda.cuMemAlloc(input_size * elem)
        )
        self._device_output = _cuda_call(
            cuda.cuMemAlloc(output_size * elem)
        )
        self._device_input_ptr = int(self._device_input)
        self._device_output_ptr = int(self._device_output)

        err, stream = cuda.cuStreamCreate(0)
        _cuda_check(err)
        self._stream_handle = int(stream)
        logger.info(
            "Buffers: input=%.0fKB output=%.0fKB",
            input_size * elem / 1024, output_size * elem / 1024,
        )

    def _set_tensor_addresses(self) -> None:
        """Bind device buffers to the TRT tensor names (TRT-10 tensor API)."""
        for name in self.input_names:
            self.context.set_tensor_address(name, self._device_input_ptr)
        for name in self.output_names:
            self.context.set_tensor_address(name, self._device_output_ptr)

    # ------------------------------------------------------- class names
    def _resolve_class_names(self, data_yaml) -> None:
        if self.class_names is not None:
            return
        if data_yaml is not None:
            try:
                parsed = _class_names_from_data_yaml(data_yaml)
                if parsed:
                    self.class_names = parsed
                    logger.info("Class names from data.yaml: %s", data_yaml)
                    return
            except Exception as e:
                logger.warning("Could not read class names from %s: %s",
                               data_yaml, e)
        # Fallback for the project's 12-class COCO subset.
        self.class_names = {i: f"class_{i}" for i in range(12)}

    # ------------------------------------------------------------- warmup
    def _warmup(self, n_iters: int = 3) -> None:
        """Exercise the real deployed path. Raise on first-call failure (a
        real config/shape error), warn+break on later hiccups — mirrors
        :meth:`OpenVINOEngine._warmup`."""
        logger.info("Warmup (TRT JIT-loads kernels on first call)...")
        dummy = np.random.rand(1, 3, self.imgsz, self.imgsz).astype(np.float32)
        try:
            self._forward(dummy)
        except Exception as e:
            raise RuntimeError(
                f"TensorRT warmup failed for {self.model_path} with input "
                f"shape {dummy.shape}: {e}. Check --imgsz matches the engine "
                f"and that the engine was built for this GPU."
            ) from e
        for _ in range(n_iters - 1):
            try:
                self._forward(dummy)
            except Exception as e:  # pragma: no cover — non-fatal warmup hiccup
                logger.warning("Non-first warmup call failed (ignored): %s", e)
                break

    # ------------------------------------------------------------- forward
    def _forward(self, batch_np: np.ndarray) -> np.ndarray:
        """Run one forward pass; return the raw ``[bs, 4+nc, N]`` head output.

        The whole H2D -> execute_async_v3 -> D2H window runs under the retained
        primary context (pushed/popped per thread) on the engine's own stream,
        then synchronised — TRT enqueues async like ORT-CUDA, so the sync is
        what makes the timing real.
        """
        bs = int(batch_np.shape[0])
        self.context.set_input_shape(
            self.input_names[0],
            [bs, self.channels, self.height, self.width],
        )
        self._set_tensor_addresses()

        flat = np.ascontiguousarray(batch_np, dtype=np.float32).ravel()
        if flat.size > self._host_input.size:
            # Grow the host staging buffer if a caller exceeded max_batch.
            self._host_input = np.empty(flat.size, dtype=np.float32)
        np.copyto(self._host_input[: flat.size], flat)

        _cuda_call(cuda.cuCtxPushCurrent(self._cuda_ctx))
        try:
            _cuda_call(cuda.cuMemcpyHtoDAsync(
                self._device_input, self._host_input.ctypes.data,
                flat.nbytes, self._stream_handle,
            ))
            self.context.execute_async_v3(self._stream_handle)
            out_shape = self.context.get_tensor_shape(self.output_names[0])
            out_bytes = (
                int(np.prod(out_shape)) * np.dtype(np.float32).itemsize
            )
            _cuda_call(cuda.cuMemcpyDtoHAsync(
                self._host_output.ctypes.data, self._device_output,
                out_bytes, self._stream_handle,
            ))
            _cuda_call(cuda.cuStreamSynchronize(self._stream_handle))
            raw = self._host_output[: int(np.prod(out_shape))].reshape(out_shape)
        finally:
            _cuda_call(cuda.cuCtxPopCurrent())
        return raw

    def raw_forward(self, batch_np: np.ndarray) -> np.ndarray:
        """Public raw-forward alias for the consistency harness."""
        return self._forward(batch_np)

    def kernel_timed_forward(
        self, batch_np: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Forward + GPU-kernel-only latency in ms (H2D+execute+D2H window).

        Preserves the TRT project's standout metric — the GPU forward pass
        free of Python NMS / letterbox / disk I/O — so the benchmark can emit
        a kernel-only column for the ``tensorrt*`` rows alongside the
        cross-backend end-to-end number.
        """
        bs = int(batch_np.shape[0])
        self.context.set_input_shape(
            self.input_names[0],
            [bs, self.channels, self.height, self.width],
        )
        self._set_tensor_addresses()
        flat = np.ascontiguousarray(batch_np, dtype=np.float32).ravel()
        if flat.size > self._host_input.size:
            self._host_input = np.empty(flat.size, dtype=np.float32)
        np.copyto(self._host_input[: flat.size], flat)

        _cuda_call(cuda.cuCtxPushCurrent(self._cuda_ctx))
        try:
            _cuda_call(cuda.cuMemcpyHtoDAsync(
                self._device_input, self._host_input.ctypes.data,
                flat.nbytes, self._stream_handle,
            ))
            self.context.execute_async_v3(self._stream_handle)
            out_shape = self.context.get_tensor_shape(self.output_names[0])
            out_bytes = (
                int(np.prod(out_shape)) * np.dtype(np.float32).itemsize
            )
            _cuda_call(cuda.cuMemcpyDtoHAsync(
                self._host_output.ctypes.data, self._device_output,
                out_bytes, self._stream_handle,
            ))
            t0 = time.perf_counter()
            _cuda_call(cuda.cuStreamSynchronize(self._stream_handle))
            kernel_ms = (time.perf_counter() - t0) * 1000.0
            raw = self._host_output[: int(np.prod(out_shape))].reshape(out_shape)
        finally:
            _cuda_call(cuda.cuCtxPopCurrent())
        return raw, kernel_ms

    # ------------------------------------------------------- effective batch
    def _effective_batch(self, requested: int) -> int:
        """Per-forward image count used to step the batch loop.

        A TRT dynamic-shape profile accepts any count in ``1..max_batch`` — no
        zero-pad, no static-batch crash (unlike a static-batch OpenVINO IR
        whose DFL reshape constant is baked). Just cap at ``max_batch`` and
        sub-loop; the tail partial batch runs directly.
        """
        eff = max(1, min(requested, self.max_batch))
        if requested > self.max_batch:
            logger.warning(
                "--batch-size=%d exceeds the engine's max_batch=%d; "
                "stepping at %d per forward (sub-looped).",
                requested, self.max_batch, eff,
            )
        return eff

    # ------------------------------------------------------------- run batch
    def _run_batch(
        self,
        data: Dict,
        conf: float,
        iou: float,
        max_det: int,
        save: bool,
        output_dir: Path,
    ) -> List[List[tuple]]:
        """Forward + post_process on one preprocessed batch.

        Shared by ``infer`` (file-backed) and ``infer_frames`` (array-backed)
        so the CLI path and the server hot path cannot drift — mirrors
        ``YOLOv8Engine._run_batch`` / ``OpenVINOEngine._run_batch``.
        """
        outputs = self._forward(data["images"].cpu().numpy())
        # Explicit check (not ``assert``) so it survives ``python -O`` — a
        # silent ndim drift would feed a mis-shaped tensor to NMS.
        if not isinstance(outputs, np.ndarray) or outputs.ndim != 3:
            raise RuntimeError(
                f"Expected a 3-D TensorRT output [bs, 4+nc, N], got "
                f"{type(outputs).__name__} ndim="
                f"{getattr(outputs, 'ndim', '?')}"
            )
        batch_dets = post_process(
            outputs=outputs,
            orig_shapes=data["orig_shapes"],
            conf_thres=conf,
            iou_thres=iou,
            imgsz=self.imgsz,
            max_det=max_det,
            ratios=data["ratios"],
            pads=data["pads"],
            # Explicit nc so NMS splits box/cls channels authoritatively —
            # the deployed path must not depend on the magnitude heuristic
            # (see src/postprocess._ensure_4nc_first). Mirrors YOLOv8Engine.
            nc=len(self.class_names) if self.class_names else None,
        )
        if save:
            for j, dets in enumerate(batch_dets):
                if not dets:
                    continue
                save_path = output_dir / f"result_{Path(data['paths'][j]).name}"
                save_annotated_image(
                    orig_img=data["orig_imgs"][j],
                    detections=dets,
                    save_path=str(save_path),
                    class_names=self.class_names,
                )
        return batch_dets

    # ------------------------------------------------------------------- infer
    def infer(
        self,
        imgs_input: Union[str, Path, List[Union[str, Path]]],
        conf: float = 0.25,
        iou: float = 0.45,
        max_imgs: int = 32,
        batch_size: int = 8,
        max_det: int = 300,
        save: bool = True,
        output_dir: Union[str, Path] = "results/predictions",
    ) -> List[List[tuple]]:
        """Run end-to-end inference with the TensorRT backend."""
        image_paths = load_images(imgs_input, max_images=max_imgs)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        eff = self._effective_batch(batch_size)
        all_results: List[List[tuple]] = []
        for i in range(0, len(image_paths), eff):
            batch_paths = image_paths[i: i + eff]
            try:
                # TRT takes host numpy and does its own H2D, so preprocess on
                # CPU (mirrors OpenVINO, which also takes host numpy).
                data = preprocess_imgs(
                    batch_paths, imgsz=self.imgsz, device="cpu", original=save,
                )
                batch_dets = self._run_batch(
                    data, conf=conf, iou=iou, max_det=max_det,
                    save=save, output_dir=output_dir,
                )
                all_results.extend(batch_dets)
                logger.info("Batch %d | Detections: %d",
                            i // eff + 1, sum(len(x) for x in batch_dets))
            except Exception:
                logger.exception("Batch inference failed (paths=%s)", batch_paths)
        return all_results

    # ----------------------------------------- infer from in-memory BGR frames
    def infer_frames(
        self,
        frames: List[np.ndarray],
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        save: bool = False,
        output_dir: Union[str, Path] = "results/predictions",
    ) -> List[List[tuple]]:
        """Inference from already-decoded BGR frames — server hot path.

        Mirrors ``OpenVINOEngine.infer_frames`` / ``YOLOv8Engine.infer_frames``:
        a decoded frame is preprocessed via ``preprocess_frames`` and run
        through the same ``_run_batch`` pipeline as the CLI's ``infer``. This
        is what lets the FastAPI server swap in the TensorRT backend without
        code changes.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        eff = self._effective_batch(len(frames))
        all_results: List[List[tuple]] = []
        for i in range(0, len(frames), eff):
            chunk = frames[i: i + eff]
            try:
                data = preprocess_frames(
                    chunk, imgsz=self.imgsz, device="cpu", original=save,
                )
                batch_dets = self._run_batch(
                    data, conf=conf, iou=iou, max_det=max_det,
                    save=save, output_dir=output_dir,
                )
                all_results.extend(batch_dets)
                logger.info("Frame batch %d | Detections: %d",
                            i // eff + 1, sum(len(x) for x in batch_dets))
            except Exception:
                logger.exception("Frame batch inference failed (offset=%d, n=%d)",
                                 i, len(chunk))
        return all_results

    # ─── Lifecycle ─────────────────────────────────────────────────────────
    def release(self) -> None:
        """Release every CUDA resource held by this engine, in deterministic
        order (stream -> device buffers -> primary context). Errors are
        swallowed so ``release()`` is safe to call multiple times and from
        ``__exit__`` on exception paths — mirrors the standalone engine."""
        if not _TRT_AVAILABLE:
            return
        try:
            try:
                _cuda_call(cuda.cuCtxPushCurrent(self._cuda_ctx))
            except Exception:
                pass
            try:
                if getattr(self, "_stream_handle", None):
                    _cuda_call(cuda.cuStreamDestroy(int(self._stream_handle)))
            except Exception:
                pass
            try:
                if getattr(self, "_device_input", None):
                    _cuda_call(cuda.cuMemFree(self._device_input))
            except Exception:
                pass
            try:
                if getattr(self, "_device_output", None):
                    _cuda_call(cuda.cuMemFree(self._device_output))
            except Exception:
                pass
            try:
                _cuda_call(cuda.cuCtxPopCurrent())
            except Exception:
                pass
            try:
                if getattr(self, "_device", None):
                    _cuda_call(cuda.cuDevicePrimaryCtxRelease(self._device))
            except Exception:
                pass
        finally:
            self._stream_handle = None
            self._device_input = None
            self._device_output = None
            self._cuda_ctx = None
            self._device = None
            self.engine = None
            self.context = None
            logger.info("TensorRT engine released")

    def close(self) -> None:
        """Back-compat alias for :meth:`release`."""
        self.release()

    def __enter__(self) -> "TensorRTEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
