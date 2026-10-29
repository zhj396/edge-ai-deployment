"""TensorRT inference backend for YOLOv8s.

Implements ``infer`` / ``infer_frames`` / ``_run_batch(nc=)``; consumes the
shared ``src.preprocess`` + ``src.postprocess`` core. TensorRT is GPU-only
and an *optional* dep; the module imports cleanly without the ``tensorrt`` /
``cuda-python`` wheels so the model-free test suite stays green.

TensorRT 10.x API contract used by this engine:

* **Tensor-name API only.** ``set_tensor_address(name, ptr)`` +
  ``execute_async_v3(stream)`` (the TRT 10 tensor-name API).
* **``cuda.bindings.driver`` "lean" bindings (CUDA 12.x).** Every Driver API
  call returns a ``(CUresult, *values)`` tuple, normalised through
  :func:`_cuda_call` / :func:`_cuda_check`; :mod:`src.tensorrt_build` uses
  the same dialect.
* **Context-managed.** The driver's primary context is adopted via
  ``cuDevicePrimaryCtxRetain`` and pushed/popped around every
  CUDA-touching region (init, forward, release). The engine targets
  single-threaded use: one execution context, one stream, and the staging
  buffers are shared across forwards. The class defines no
  destructor: teardown is explicit through ``release()`` / ``close()`` /
  the context-manager protocol, in stream -> device buffers -> TRT
  objects -> primary context order.
* **Dynamic batch.** ``set_input_shape`` per forward sets the actual batch
  count, so any count in ``1..max_batch`` runs directly. A static engine
  (no optimization profile) steps at batch 1.
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
# names come from data.yaml via ``_class_names_from_data_yaml``.
from .openvino_engine import _class_names_from_data_yaml

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional-dep guard — tensorrt + cuda-python are NOT in the base pin set.
# Importing without the wheels sets _TRT_AVAILABLE=False; constructing
# TensorRTEngine then raises ImportError at construction.
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
# cuda.bindings.driver "lean" helpers — the single source of truth for the
# tuple-return style used by the engine; :mod:`src.tensorrt_build` imports
# these helpers. Lean bindings return (CUresult, *values) for every call.
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
    CUresult. Most ``cuda.<fn>(...)`` call sites go through this function;
    ``_cuda_check`` covers the call sites that destructure the returned
    tuple themselves.
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


def set_cuda_visible_devices(device_id: int) -> Optional[str]:
    """Pin ``CUDA_VISIBLE_DEVICES`` to one device — BEFORE any CUDA init.

    Process-level and effective only before ``cuInit`` (or any other CUDA
    initialization, including another engine built on a different GPU id in
    the same process): CUDA reads the variable once at init, so setting it
    afterwards has no effect and a second engine with a different ``device``
    would silently stay on the already-initialized device set. Returns the
    previous value (``None`` when the variable was unset) so the caller can
    restore it via :func:`restore_cuda_visible_devices` once its own ``cuInit``
    has read the pin.
    """
    if device_id < 0:
        raise ValueError(f"device_id must be >= 0, got {device_id}")
    target = str(device_id)
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    if prev != target:
        os.environ["CUDA_VISIBLE_DEVICES"] = target
    return prev


def restore_cuda_visible_devices(prev: Optional[str]) -> None:
    """Restore the value captured by :func:`set_cuda_visible_devices`.

    ``prev=None`` removes the variable (the state before the pin). This only
    affects code that reads the environment afterwards — child processes and
    later libraries inspecting the env — since the already-initialized CUDA
    driver state in this process is not re-enumerated.
    """
    if prev is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = prev


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
        Caller-side upper bound the engine is trusted with (default 8). The
        serialized engine's own optimization-profile max caps it further, and
        a static engine (no profile) runs at batch 1; forwards are capped at
        the effective value.
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
        if self.max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")
        self._batch_mismatch_warned = False
        self.class_names = class_names

        device_id = self._resolve_device_id(device)
        # CUDA_VISIBLE_DEVICES must be set before cuInit so the runtime only
        # sees the requested device. The previous value is restored once this
        # cuInit has read the pin (also on the failure paths below): the pin
        # exists for this engine's own device enumeration, and the process
        # env keeps the caller's value so child processes (e.g. the ort_cpp
        # driver) and later env readers are not restricted to the pinned GPU.
        prev_cvd = set_cuda_visible_devices(device_id)
        try:
            _cuda_call(cuda.cuInit(0))
            # Post-init visibility check: the pin takes effect only when this
            # cuInit is the process's first CUDA initialization. More than one
            # visible device under the pin means an earlier backend (a CUDA-EP
            # ORT session, a torch model) initialized CUDA first and the pin is
            # inert — ordinal 0 then resolves to physical GPU 0, so record that
            # id as the device this engine actually executes on.
            visible = int(_cuda_call(cuda.cuDeviceGetCount()))
            if device_id != 0 and visible > 1:
                logger.warning(
                    "CUDA was already initialized in this process, so the "
                    "CUDA_VISIBLE_DEVICES=%d pin is inert; the engine runs on "
                    "physical GPU 0. Construct the TensorRT engine before any "
                    "other CUDA backend (or in a fresh process) to honor the pin.",
                    device_id,
                )
                device_id = 0
            elif device_id != 0:
                logger.warning(
                    "CUDA_VISIBLE_DEVICES=%d took effect at this process's "
                    "first CUDA init: the CUDA device set stays narrowed to "
                    "this GPU for the process lifetime (cuInit reads the "
                    "variable once), so CUDA backends constructed later in "
                    "this process see only this device.",
                    device_id,
                )
            self.device_id = device_id
            # After cuInit under the pin, CUDA re-enumerates the filtered set:
            # the requested physical GPU is ordinal 0 (cuDeviceGet(device_id)
            # would address the (device_id+1)-th visible device and raise
            # CUDA_ERROR_INVALID_DEVICE on the last GPU).
            self._device = _cuda_call(cuda.cuDeviceGet(0))
        finally:
            restore_cuda_visible_devices(prev_cvd)
        # Adopt the driver's primary context: it is made current here so
        # engine deserialization, execution-context creation, and
        # buffer/stream allocation run under it (CUDA objects belong to the
        # context they were created on), then pushed/popped around every
        # forward.
        self._cuda_ctx = _cuda_call(cuda.cuDevicePrimaryCtxRetain(self._device))

        try:
            _cuda_call(cuda.cuCtxPushCurrent(self._cuda_ctx))
        except BaseException:
            # The push failed, so the failure path below would never pop this
            # context — release the retention here.
            try:
                _cuda_call(cuda.cuDevicePrimaryCtxRelease(self._device))
            except Exception:
                pass
            self._cuda_ctx = None
            self._device = None
            raise
        pushed = True
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Engine not found: {self.model_path}")
            self._load_engine()
            self._resolve_class_names(data_yaml)
            self._warmup()
        except BaseException:
            # Construction failed: free the partial allocations and drop the
            # TRT objects while the context is still current (they own GPU
            # resources tied to it), then pop and drop the retention in the
            # same pop-then-release order release() uses, so a failed
            # construction leaves neither device buffers nor the
            # primary-context reference behind.
            try:
                self._free_buffers()
            except Exception:
                pass
            self.context = None
            self.engine = None
            if pushed:
                try:
                    _cuda_call(cuda.cuCtxPopCurrent())
                except Exception:
                    pass
                pushed = False
            try:
                _cuda_call(cuda.cuDevicePrimaryCtxRelease(self._device))
            except Exception:
                pass
            # Null the released/owned refs so a later release() (e.g. from
            # __exit__ on the exception) is a no-op: a dead context must not
            # be pushed again, and the primary context must not be released
            # a second time.
            self._cuda_ctx = None
            self._device = None
            self.engine = None
            self.context = None
            raise
        finally:
            if pushed:
                try:
                    _cuda_call(cuda.cuCtxPopCurrent())
                except Exception:
                    pass
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

        if len(self.input_names) != 1:
            raise RuntimeError(
                f"Engine has {len(self.input_names)} input tensors "
                f"{self.input_names}; this engine binds exactly one image "
                "input (the device-input buffer is shared across inputs)."
            )
        for name in self.input_names + self.output_names:
            dtype = self.engine.get_tensor_dtype(name)
            if dtype != trt.float32:
                raise RuntimeError(
                    f"Tensor {name} has dtype {dtype}; this engine supports "
                    "float32 I/O only (FP16 stays internal to the build; a "
                    "narrower I/O dtype would be sized/copied with the "
                    "float32 itemsize)."
                )

        input_shape = self.engine.get_tensor_shape(self.input_names[0])
        if any(int(d) < 0 for d in input_shape[1:]):
            raise RuntimeError(
                f"Engine input {self.input_names[0]} has dynamic spatial dims "
                f"{tuple(input_shape)}: this engine only supports a dynamic "
                "batch dim (build with --static or a fixed imgsz)."
            )
        self.channels = int(input_shape[1]) if len(input_shape) > 1 else 3
        self.height = int(input_shape[2]) if len(input_shape) > 2 else 640
        self.width = int(input_shape[3]) if len(input_shape) > 3 else 640

        # Resolve the effective batch bound from the serialized engine: the
        # optimization profile's max for a dynamic engine, 1 for a static
        # engine (the batch=1 build route carries no profile).
        if self.engine.num_optimization_profiles:
            profile_max = self.engine.get_tensor_profile_shape(
                self.input_names[0], 0
            )[2][0]
            self.max_batch = min(self.max_batch, int(profile_max))
        else:
            if int(input_shape[0]) != 1:
                raise RuntimeError(
                    f"Static engine input batch is {input_shape[0]}; only a "
                    "batch=1 static engine is supported."
                )
            self.max_batch = 1
            logger.info(
                "Static-batch engine (no optimization profile); forwards "
                "step at 1 image."
            )

        self._allocate_buffers()

    # -------------------------------------------------------- allocate buffers
    def _allocate_buffers(self) -> None:
        """Allocate per-output host/device buffers + a CUDA stream.

        Each output tensor gets its own device buffer: every output name
        needs a distinct address for ``execute_async_v3``. Sizes come from
        each tensor's shape, with dynamic dims resolved to the optimization
        profile's max shape, so any in-profile batch fits the pre-allocated
        buffers. The detect head is the first output tensor of a YOLOv8
        export; ``_forward`` copies back and returns that one, so additional
        outputs (if any) get device buffers but are not consumed.
        """
        input_size = (
            self.max_batch * self.channels * self.height * self.width
        )

        def _max_shape(name, shape):
            """Concrete allocation shape for one tensor.

            Dynamic dims resolve to the optimization profile's max shape —
            the exact bound the engine can emit, independent of imgsz or the
            class count.

            Note: ``get_tensor_profile_shape`` is documented for INPUT
            tensors; calling it on a dynamic OUTPUT tensor (the YOLOv8 head
            is ``[-1, 4+nc, 8400]``) is an assumption this engine makes — it
            is expected to work because TRT binds whole profiles to the
            I/O tensor set, but it is not covered by the TRT docs.
            """
            concrete = [int(d) for d in shape]
            if any(d == -1 for d in concrete):
                if not self.engine.num_optimization_profiles:
                    raise RuntimeError(
                        f"Tensor {name} has dynamic dims {tuple(shape)} but "
                        "the engine carries no optimization profile."
                    )
                _min, _opt, mx = self.engine.get_tensor_profile_shape(name, 0)
                concrete = [int(d) for d in mx]
            return concrete

        elem = np.dtype(np.float32).itemsize
        self._host_input = np.empty(input_size, dtype=np.float32)

        self._host_outputs: List[np.ndarray] = []
        self._device_outputs = []
        self._device_output_ptrs: List[int] = []
        for name in self.output_names:
            shape = self.engine.get_tensor_shape(name)
            concrete = _max_shape(name, shape)
            size = int(np.prod(concrete))
            logger.info("Output %s shape %s (concrete max %s)",
                        name, shape, concrete)
            self._host_outputs.append(np.empty(size, dtype=np.float32))
            dev = _cuda_call(cuda.cuMemAlloc(size * elem))
            self._device_outputs.append(dev)
            self._device_output_ptrs.append(int(dev))

        self._device_input = _cuda_call(
            cuda.cuMemAlloc(input_size * elem)
        )
        self._device_input_ptr = int(self._device_input)

        err, stream = cuda.cuStreamCreate(0)
        _cuda_check(err)
        self._stream_handle = int(stream)
        logger.info(
            "Buffers: input=%.0fKB output=%.0fKB (%d tensors)",
            input_size * elem / 1024,
            sum(h.nbytes for h in self._host_outputs) / 1024,
            len(self._host_outputs),
        )

    def _set_tensor_addresses(self) -> None:
        """Bind device buffers to the TRT tensor names (TRT-10 tensor API)."""
        for name in self.input_names:
            self.context.set_tensor_address(name, self._device_input_ptr)
        for name, ptr in zip(self.output_names, self._device_output_ptrs):
            self.context.set_tensor_address(name, ptr)

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
        real config/shape error), warn+break on later hiccups."""
        logger.info("Warmup (TRT JIT-loads kernels on first call)...")
        dummy = np.random.rand(
            1, self.channels, self.imgsz, self.imgsz
        ).astype(np.float32)
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
    def _forward(
        self, batch_np: np.ndarray, timed: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """Run one forward pass; return the raw ``[bs, 4+nc, N]`` head output.

        The H2D copy, ``execute_async_v3``, and the D2H copy are enqueued on
        the engine's stream under the retained primary context (pushed/popped
        around the pass); ``cuStreamSynchronize`` makes the forward synchronous
        to the caller.

        With ``timed=True`` the timer brackets the stream synchronization:
        the returned ``kernel_ms`` is the GPU-side forward window observed at
        sync (GPU work already retired before the timer starts does not
        re-appear in it). The kernels and the D2H copy are the window's
        regular content; the H2D copy is enqueued before the timer starts
        and, if it is still pending at that point, its tail lands inside
        the window. The window excludes Python
        NMS / letterbox / disk I/O, so it isolates the GPU forward from
        the surrounding host-side work. The return is then
        ``(output, kernel_ms)``.
        """
        bs = int(batch_np.shape[0])
        if (
            batch_np.ndim != 4
            or tuple(batch_np.shape[1:])
            != (self.channels, self.height, self.width)
        ):
            raise ValueError(
                f"Input batch shape {tuple(batch_np.shape)} does not match "
                f"the engine input (N, {self.channels}, {self.height}, "
                f"{self.width}); pass a matching --imgsz or rebuild the "
                "engine at this size. set_input_shape carries the engine's "
                "own spatial dims, so a mismatched input would be H2D-copied "
                "short and executed against a stale buffer tail."
            )
        if not self.context.set_input_shape(
            self.input_names[0],
            [bs, self.channels, self.height, self.width],
        ):
            raise ValueError(
                f"Input shape {[bs, self.channels, self.height, self.width]} "
                f"was rejected by the engine (max_batch {self.max_batch})."
            )
        self._set_tensor_addresses()

        flat = np.ascontiguousarray(batch_np, dtype=np.float32).ravel()
        if flat.size > self._host_input.size:
            # The pre-allocated device buffer holds max_batch inputs.
            raise ValueError(
                f"batch {bs} exceeds the engine's max_batch={self.max_batch}; "
                "lower the batch size or rebuild the engine with a larger "
                "max_batch profile"
            )
        np.copyto(self._host_input[: flat.size], flat)

        pushed = False
        try:
            _cuda_call(cuda.cuCtxPushCurrent(self._cuda_ctx))
            pushed = True
            _cuda_call(cuda.cuMemcpyHtoDAsync(
                self._device_input, self._host_input.ctypes.data,
                flat.nbytes, self._stream_handle,
            ))
            if not self.context.execute_async_v3(self._stream_handle):
                raise RuntimeError(
                    "execute_async_v3 failed to enqueue the forward"
                )
            out_shape = self.context.get_tensor_shape(self.output_names[0])
            if any(int(d) < 0 for d in out_shape):
                # Same contract as the input-side shape check above: the D2H
                # copy and the reshape need concrete dims.
                raise RuntimeError(
                    f"Output tensor {self.output_names[0]} still carries "
                    f"dynamic dims {list(out_shape)} after set_input_shape; "
                    "the D2H copy cannot be sized. Rebuild the engine from "
                    "an ONNX whose output shape is input-determined."
                )
            out_bytes = (
                int(np.prod(out_shape)) * np.dtype(np.float32).itemsize
            )
            _cuda_call(cuda.cuMemcpyDtoHAsync(
                self._host_outputs[0].ctypes.data, self._device_outputs[0],
                out_bytes, self._stream_handle,
            ))
            t0 = time.perf_counter() if timed else None
            _cuda_call(cuda.cuStreamSynchronize(self._stream_handle))
            kernel_ms = (time.perf_counter() - t0) * 1000.0 if timed else None
            raw = self._host_outputs[0][: int(np.prod(out_shape))].reshape(out_shape)
        finally:
            if pushed:
                # Pop only the context pushed above; a failed push must not
                # unwind whatever context the calling thread already has.
                # Swallow a failed pop so it cannot mask the body's original
                # exception (same guard as release()).
                try:
                    _cuda_call(cuda.cuCtxPopCurrent())
                except Exception:
                    pass
        return (raw, kernel_ms) if timed else raw

    def raw_forward(self, batch_np: np.ndarray) -> np.ndarray:
        """Public raw-forward alias for the consistency harness."""
        return self._forward(batch_np)  # type: ignore[return-value]

    def kernel_timed_forward(
        self, batch_np: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Forward + GPU-side forward latency in ms; see :meth:`_forward`."""
        out = self._forward(batch_np, timed=True)
        return out  # type: ignore[return-value]

    # ------------------------------------------------------- effective batch
    def _effective_batch(self, requested: int) -> int:
        """Per-forward image count used to step the batch loop.

        Cap the requested count at ``max_batch``; the ``infer`` /
        ``infer_frames`` loops step over forwards at that count and the tail
        partial batch runs directly. A dynamic-batch profile accepts any
        count in ``1..max_batch``; a static engine steps at 1.
        """
        eff = max(1, min(requested, self.max_batch))
        if requested > self.max_batch:
            # Warn once per engine instance: the server path calls this per
            # request, and the cap is a property of the engine, not of the
            # request.
            if not self._batch_mismatch_warned:
                self._batch_mismatch_warned = True
                logger.warning(
                    "Requested per-forward batch %d exceeds the engine's "
                    "max_batch=%d; stepping at %d per forward (sub-looped).",
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

        Single body shared by ``infer`` (file-backed) and ``infer_frames``
        (array-backed).
        """
        outputs = self._forward(data["images"].cpu().numpy())
        # Validate the output shape before NMS.
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
            # nc is known from class_names, so pass it explicitly and NMS
            # splits box/cls channels by the class count.
            nc=len(self.class_names) if self.class_names else None,
        )
        if save:
            for j, dets in enumerate(batch_dets):
                if not dets:
                    continue
                save_path = output_dir / f"result_{Path(data['paths'][j]).name}"
                if not save_path.suffix:
                    # Synthetic frame paths carry no extension; cv2.imwrite
                    # picks the encoder from the file extension.
                    save_path = save_path.with_suffix(".jpg")
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
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)

        eff = self._effective_batch(batch_size)
        all_results: List[List[tuple]] = []
        for i in range(0, len(image_paths), eff):
            batch_paths = image_paths[i: i + eff]
            try:
                # TRT does its own H2D, so preprocess on CPU.
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
        """Inference from already-decoded BGR frames.

        Feeds decoded frames through ``preprocess_frames`` →
        ``_run_batch``.
        """
        # Parity with YOLOv8Engine.infer_frames / OpenVINOEngine.infer_frames:
        # an empty frame list is a caller error, not an empty-but-successful
        # result — the server hot path must see the same behavior from any
        # engine.
        if not frames:
            raise ValueError("Frame list cannot be empty")
        output_dir = Path(output_dir)
        if save:
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
    def _free_buffers(self) -> None:
        """Destroy the stream and free every device buffer (idempotent).

        Runs with the primary context current (``release()`` pushes it; the
        constructor-failure path calls it inside its own pushed region).
        """
        try:
            if getattr(self, "_stream_handle", None) is not None:
                _cuda_call(cuda.cuStreamDestroy(int(self._stream_handle)))
        except Exception:
            pass
        self._stream_handle = None
        try:
            if getattr(self, "_device_input", None) is not None:
                _cuda_call(cuda.cuMemFree(self._device_input))
        except Exception:
            pass
        self._device_input = None
        for dev in getattr(self, "_device_outputs", None) or []:
            try:
                _cuda_call(cuda.cuMemFree(dev))
            except Exception:
                pass
        self._device_outputs = []
        self._device_output_ptrs = []
        self._host_outputs = []

    def release(self) -> None:
        """Release every resource held by this engine, in deterministic order
        (stream -> device buffers -> TRT objects -> primary context). Errors
        are swallowed so ``release()`` is safe to call multiple times and
        from ``__exit__`` on exception paths."""
        if not _TRT_AVAILABLE:
            return
        pushed = False
        try:
            try:
                _cuda_call(cuda.cuCtxPushCurrent(self._cuda_ctx))
                pushed = True
            except Exception:
                pass
            self._free_buffers()
            # Drop the TRT objects while the primary context is still
            # current: the execution context and the deserialized engine own
            # GPU resources tied to it, so they must not outlive it.
            self.context = None
            self.engine = None
            if pushed:
                # Pop only the context pushed above; a failed push must not
                # unwind whatever context the calling thread already has.
                try:
                    _cuda_call(cuda.cuCtxPopCurrent())
                except Exception:
                    pass
            try:
                if getattr(self, "_device", None) is not None:
                    _cuda_call(cuda.cuDevicePrimaryCtxRelease(self._device))
            except Exception:
                pass
        finally:
            self._stream_handle = None
            self._device_input = None
            self._device_outputs = None
            self._device_output_ptrs = None
            self._host_outputs = None
            self._cuda_ctx = None
            self._device = None
            self.engine = None
            self.context = None
            logger.info("TensorRT engine released")

    def close(self) -> None:
        """Alias for :meth:`release`."""
        self.release()

    def __enter__(self) -> "TensorRTEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
