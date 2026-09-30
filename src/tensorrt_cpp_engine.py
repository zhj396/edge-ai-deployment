"""In-process TensorRT C++ backend for YOLOv8s (pybind11 accelerator).

Mirrors :class:`src.tensorrt_engine.TensorRTEngine`'s public API EXACTLY
(``__init__`` / ``raw_forward`` / ``kernel_timed_forward`` / ``_effective_batch``
/ ``_run_batch`` / ``infer`` / ``infer_frames`` / ``release`` / ``__enter__`` /
``__exit__`` / ``close``), but delegates the GPU forward to the loaded
``_trt_cpp`` pybind11 module (CUDA Runtime API + TRT-10 tensor-name API in C++,
no subprocess). Preprocess/postprocess stay Python — they reuse the shared
:mod:`src.preprocess` + :mod:`src.postprocess` core like every backend; the C++
module is a thin forward-only accelerator that deserialises a ``.engine`` and
runs ``execute_async_v3``, handing the raw ``[bs, 4+nc, 8400]`` head back.

Why this exists (Option B, in-process — not a subprocess): so ``trt_cpp``
compares **apples-to-apples** with the Python ``tensorrt:`` path on the same
metric surface — native mAP (via :func:`utils.map_eval.evaluate_map` driving
``infer``) + ``kernel_latency_ms`` (the H2D+execute+D2H window free of Python
NMS). That makes the C++-binding-vs-cuda-python overhead comparison the real
value of this backend; a subprocess + ``.npy`` exchange (the ``ort_cpp``
template) would lack both and leave the comparison limping.

Optional-dep guard mirrors :mod:`src.tensorrt_engine`: importing this module
without the built ``_trt_cpp`` .so/.pyd sets ``_TRT_CPP_AVAILABLE=False`` (probed
via :func:`src.benchmark.resolve_trt_cpp_module`, never a top-level import);
constructing ``TensorRTEngineCpp`` then raises ``ImportError`` with the build
hint (never ``AttributeError``). The model-free test suite stays green on a
CPU/CI box without TensorRT/CUDA/pybind11.

CUDA context note: the C++ module uses the CUDA **Runtime** API
(``cudaMalloc`` / ``cudaMemcpyAsync`` / ``cudaStreamSynchronize``), which
auto-manages the primary context — no ``cuCtxPushCurrent``/``cuCtxPopCurrent``
(the Python engine's push/pop is a Driver-API artifact via cuda-python lean
bindings). The two backends share the underlying primary context (refcounted,
sequential); the benchmark always calls ``release()`` between backends, so
only one is live at a time. Do not construct both simultaneously.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from utils import get_logger, load_images, save_annotated_image
from . import post_process, preprocess_imgs, preprocess_frames
# A serialized TRT engine carries no ultralytics "names" metadata, so class
# names come from data.yaml — same as the Python TRT path; reuse the OpenVINO
# engine's pure-Python reader (same YOLO names field, same list/dict
# normalization) rather than forking it.
from .openvino_engine import _class_names_from_data_yaml

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional-dep guard — the _trt_cpp pybind11 module is NOT built by default
# (BUILD_TRT_CPP=OFF; requires TensorRT + CUDA + pybind11). Resolved LAZILY on
# the first trt_cpp_available() call (probes resolve_trt_cpp_module, never a
# top-level import) so importing this module / constructing the wrapper
# without the .so/.pyd built does not blow up — only an actual forward call
# resolves and may raise the helpful error. Mirrors the Python TRT guard.
# ---------------------------------------------------------------------------
_TRT_CPP_AVAILABLE: Optional[bool] = None


def trt_cpp_available() -> bool:
    """True if the built ``_trt_cpp`` pybind11 module loads on this box."""
    global _TRT_CPP_AVAILABLE
    if _TRT_CPP_AVAILABLE is None:
        from .benchmark import resolve_trt_cpp_module
        try:
            resolve_trt_cpp_module()
            _TRT_CPP_AVAILABLE = True
        except Exception:
            _TRT_CPP_AVAILABLE = False
    return _TRT_CPP_AVAILABLE


class TensorRTEngineCpp:
    """C++ pybind11-backed TensorRT engine.

    Parameters mirror :class:`src.tensorrt_engine.TensorRTEngine` exactly so
    the CLI, the benchmark harness, the consistency harness, and the server
    hot path plug in unchanged. See that class's docstring for the per-param
    semantics; only the forward implementation differs (C++ Runtime API via
    the ``_trt_cpp`` module, vs Python cuda-python Driver API).
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
        if not trt_cpp_available():
            raise ImportError(
                "trt_cpp pybind11 module not built. Build it first: "
                "`cmake -S cpp -B cpp/build -DBUILD_TRT_CPP=ON && "
                "cmake --build cpp/build --target _trt_cpp` (or set "
                "TRT_CPP_PATH). Requires TensorRT + CUDA + pybind11. "
                "See docs/TENSORRT.md."
            )

        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.max_batch = int(max_batch)
        self.class_names = class_names
        self.device_id = self._resolve_device_id(device)

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Engine not found: {self.model_path}")

        # Load the pybind11 module + construct the C++ TrtEngine. The C++
        # constructor does cudaSetDevice + createInferRuntime +
        # deserializeCudaEngine + createExecutionContext + allocate_buffers +
        # setTensorAddress; errors surface as RuntimeError (pybind11
        # translates std::runtime_error).
        from .benchmark import resolve_trt_cpp_module
        self._module = resolve_trt_cpp_module()
        self._engine = self._module.TrtEngine(
            engine_path=self.model_path, device=self.device_id,
            imgsz=self.imgsz, max_batch=self.max_batch,
        )
        self._resolve_class_names(data_yaml)
        self._warmup()
        logger.info(
            "trt_cpp engine initialized | %s | max_batch=%d | device=%d",
            Path(self.model_path).name, self.max_batch, self.device_id,
        )

    # ----------------------------------------------------------------- device
    @staticmethod
    def _resolve_device_id(device: Union[int, str]) -> int:
        """Coerce ``int | "cuda" | "cuda:N"`` to a numeric GPU id.

        Identical to :meth:`TensorRTEngine._resolve_device_id`.
        """
        if isinstance(device, int) and not isinstance(device, bool):
            return device
        s = str(device).lower()
        if s == "cpu":
            raise ValueError("TensorRT is GPU-only; device='cpu' is invalid")
        if s in ("cuda", "cuda:0"):
            return 0
        if s.startswith("cuda:"):
            return int(s.split(":", 1)[1])
        return int(s)

    # ---------------------------------------------------------- class names
    def _resolve_class_names(self, data_yaml) -> None:
        """Identical to :meth:`TensorRTEngine._resolve_class_names`."""
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

    # -------------------------------------------------------------- warmup
    def _warmup(self, n_iters: int = 3) -> None:
        """Exercise the real deployed path (mirrors TensorRTEngine._warmup).

        TRT JIT-loads kernels on the first call; a warmup failure usually means
        the engine was built for a different GPU/imgsz than the runtime."""
        logger.info("Warmup (TRT JIT-loads kernels on first call)...")
        dummy = np.random.rand(1, 3, self.imgsz, self.imgsz).astype(np.float32)
        try:
            self._forward(dummy)
        except Exception as e:
            raise RuntimeError(
                f"trt_cpp warmup failed for {self.model_path} with input "
                f"shape {dummy.shape}: {e}. Check --imgsz matches the "
                f"engine and that the engine was built for this GPU."
            ) from e
        for _ in range(n_iters - 1):
            try:
                self._forward(dummy)
            except Exception as e:  # pragma: no cover — non-fatal warmup hiccup
                logger.warning("Non-first warmup call failed (ignored): %s", e)
                break

    # ------------------------------------------------------------- forward
    def _forward(self, batch_np: np.ndarray) -> np.ndarray:
        """Run one forward pass via the C++ module; return the raw
        ``[bs, 4+nc, N]`` head output as a numpy array.

        The C++ side does H2D -> execute_async_v3 -> D2H -> sync on its own
        CUDA stream (Runtime API), returning a fresh numpy array (the staging
        buffer is owned C++-side; the returned array owns its own data via a
        memcpy — no shared device-pointer across the boundary).
        """
        return self._engine.raw_forward(
            np.ascontiguousarray(batch_np, dtype=np.float32)
        )

    def raw_forward(self, batch_np: np.ndarray) -> np.ndarray:
        """Public raw-forward alias for the consistency harness."""
        return self._forward(batch_np)

    def kernel_timed_forward(
        self, batch_np: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Forward + GPU-kernel-only latency in ms (H2D+execute+D2H window).

        Mirrors :meth:`TensorRTEngine.kernel_timed_forward`: the C++ timer
        brackets only ``cudaStreamSynchronize`` (the point where all the
        enqueued async work completes), so the reported ms is free of Python
        NMS / letterbox / disk I/O. The benchmark emits this as
        ``kernel_latency_ms`` / ``kernel_fps`` for ``trt_cpp*`` rows.
        """
        out, ms = self._engine.kernel_timed_forward(
            np.ascontiguousarray(batch_np, dtype=np.float32)
        )
        return out, float(ms)

    # ------------------------------------------------------- effective batch
    def _effective_batch(self, requested: int) -> int:
        """Per-forward image count used to step the batch loop.

        Identical to :meth:`TensorRTEngine._effective_batch`: a TRT dynamic-
        shape profile accepts any count in ``1..max_batch`` — no zero-pad, no
        static-batch crash. Just cap at ``max_batch`` and sub-loop.
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

        Byte-for-byte parity with :meth:`TensorRTEngine._run_batch` — the only
        difference is ``self._forward`` resolves to the C++ module call above.
        The explicit ``nc=len(self.class_names)`` is invariant #9: the deployed
        path must not depend on the postprocess magnitude heuristic.
        """
        outputs = self._forward(data["images"].cpu().numpy())
        if not isinstance(outputs, np.ndarray) or outputs.ndim != 3:
            raise RuntimeError(
                f"Expected a 3-D trt_cpp output [bs, 4+nc, N], got "
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
        """Run end-to-end inference with the trt_cpp backend.

        Mirrors :meth:`TensorRTEngine.infer`: preprocess on CPU (TRT takes
        host numpy and does its own H2D), step at ``_effective_batch``, run
        the shared ``_run_batch``.
        """
        image_paths = load_images(imgs_input, max_images=max_imgs)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        eff = self._effective_batch(batch_size)
        all_results: List[List[tuple]] = []
        for i in range(0, len(image_paths), eff):
            batch_paths = image_paths[i: i + eff]
            try:
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

        Mirrors :meth:`TensorRTEngine.infer_frames`: the same ``_run_batch``
        pipeline as ``infer``, fed by ``preprocess_frames`` (array-backed, no
        disk round-trip) so the FastAPI server swaps in trt_cpp unchanged.
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
        """Release the C++ engine's CUDA resources (stream + device buffers).

        Idempotent + safe in ``finally`` / ``__exit__`` — the C++ ``release()``
        guards on a ``released`` flag and nulls all handles. The Runtime API
        primary context is auto-managed (freed at process exit), so there is no
        explicit context release here (contrast the Python path's
        ``cuDevicePrimaryCtxRelease`` — that's a Driver-API artifact).
        """
        if getattr(self, "_engine", None) is not None:
            try:
                self._engine.release()
            except Exception:
                pass
            self._engine = None
        logger.info("trt_cpp engine released")

    def close(self) -> None:
        """Back-compat alias for :meth:`release`."""
        self.release()

    def __enter__(self) -> "TensorRTEngineCpp":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
