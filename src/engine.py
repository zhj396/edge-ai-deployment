"""Unified YOLOv8 inference engine across PyTorch / ONNX FP32 / ONNX INT8.

Highlights
----------
* Single ``infer()`` API regardless of backend.
* CPU backend runs a plain numpy float32 ``session.run`` path.
* CUDA backend uses **IO Binding with DLPack** so the input tensor stays on the device (zero
host→device copy); outputs are copied back to the host once per inference via
``copy_outputs_to_cpu()``.
* Warm-up runs the deployed backend's input path (numpy for CPU, OrtValue/DLPack
for CUDA) to absorb ORT's first-call compilation, allocation, and CUDA kernel
lazy-initialization cost.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.dlpack import to_dlpack
from ultralytics import YOLO

from utils import get_logger, load_images, save_annotated_image, select_providers
from . import post_process, preprocess_imgs, preprocess_frames

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _class_names_from_session(session: ort.InferenceSession, num_classes: int = 12) -> dict:
    """Best-effort class-name extraction from an ONNX model's metadata."""
    try:
        from ast import literal_eval

        meta = session.get_modelmeta()
        names_meta = meta.custom_metadata_map.get("names")
        if names_meta:
            # Parse the metadata string with ``literal_eval`` (untrusted model content).
            parsed = literal_eval(names_meta)
            if isinstance(parsed, dict):
                return {int(k): v for k, v in parsed.items()}
            if isinstance(parsed, list):
                return {i: n for i, n in enumerate(parsed)}
    except Exception:
        pass
    # Fallback for our 12-class COCO subset
    return {i: f"class_{i}" for i in range(num_classes)}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class YOLOv8Engine:
    """Multi-backend YOLOv8 inference engine.

    Parameters
    ----------
    model_path
        Path to a ``.pt`` (PyTorch) or ``.onnx`` model.
    backend
        ``pytorch`` | ``onnx_fp32`` | ``onnx_int8``.
    imgsz
        Square input size (default 640).
    device
        ``cpu`` or ``cuda``. CUDA auto-falls back to CPU if unavailable.
    class_names
        Optional override for class names; otherwise inferred from the model metadata.
    intra_op_threads, inter_op_threads
        Thread-pool sizing for ORT (defaults: ``intra=4``, ``inter=2``, capped at core count).
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "pytorch",
        imgsz: int = 640,
        device: str = "cpu",
        class_names: Optional[Dict[int, str]] = None,
        intra_op_threads: Optional[int] = None,
        inter_op_threads: Optional[int] = None,
    ) -> None:
        self.model_path = model_path
        self.backend = backend.lower()
        self.imgsz = imgsz
        self.device = (
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.class_names = class_names
        # Defaults: ``intra=4``, ``inter=2``, capped at core count.
        ncpu = os.cpu_count() or 1
        self.intra_op_threads = (
            intra_op_threads if intra_op_threads is not None else min(4, ncpu)
        )
        self.inter_op_threads = (
            inter_op_threads
            if inter_op_threads is not None
            else (2 if self.intra_op_threads > 1 else 1)
        )

        self.model: Optional[YOLO] = None
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: Optional[List[str]] = None

        self._load_model()
        self._warmup()
        logger.info("%s engine initialized", self.backend.upper())

    # load
    def _load_model(self) -> None:
        if self.backend == "pytorch":
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            if self.class_names is None:
                self.class_names = self.model.names

        elif self.backend.startswith("onnx"):
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            if self.intra_op_threads is not None:
                sess_options.intra_op_num_threads = self.intra_op_threads
            if self.inter_op_threads is not None:
                sess_options.inter_op_num_threads = self.inter_op_threads

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=select_providers(self.device),
            )

            # Warn when CUDA was requested but is not active
            active = self.session.get_providers()

            if self.device == "cuda" and "CUDAExecutionProvider" not in active:
                logger.warning(
                    "CUDAExecutionProvider requested but not active; "
                    "providers=%s",
                    active,
                )

            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            if self.class_names is None:
                self.class_names = _class_names_from_session(self.session)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    # warmup
    def _warmup(self, n_iters: int = 3) -> None:
        """Run a few inferences through the deployed per-backend/per-device path.

        Absorbs ORT first-call work (CUDA arena allocation, cuDNN lazy-load,
        JIT compilation). On CUDA the warmup binds a real CUDA tensor via
        DLPack, matching ``_forward``'s CUDA branch.
        """
        logger.info("Warmup...")
        if self.backend == "pytorch":
            dummy = torch.rand(1, 3, self.imgsz, self.imgsz, device=self.device)
            for _ in range(n_iters):
                try:
                    with torch.no_grad():
                        self.model.model(dummy)
                except Exception as e:
                    logger.warning("Warmup iteration failed: %s", e)
                    break
            return

        if self.device == "cuda" and torch.cuda.is_available():
            # CUDA warmup: DLPack OrtValue input, device-bound outputs.
            dummy = torch.rand(1, 3, self.imgsz, self.imgsz, device="cuda")
            for _ in range(n_iters):
                try:
                    io_binding = self.session.io_binding()
                    ort_input = ort.OrtValue.from_dlpack(to_dlpack(dummy.contiguous()))
                    io_binding.bind_ortvalue_input(self.input_name, ort_input)
                    for out in self.output_names:
                        io_binding.bind_output(out, "cuda")
                    # Same cross-stream barrier as ``_forward``'s CUDA branch.
                    torch.cuda.current_stream().synchronize()
                    self.session.run_with_iobinding(io_binding)
                    _ = io_binding.copy_outputs_to_cpu()
                except Exception as e:
                    logger.warning("Warmup iteration failed: %s", e)
                    break
        else:
            dummy = np.random.rand(1, 3, self.imgsz, self.imgsz).astype(np.float32)
            for _ in range(n_iters):
                try:
                    self.session.run(self.output_names, {self.input_name: dummy})
                except Exception as e:
                    logger.warning("Warmup iteration failed: %s", e)
                    break

    # forward
    def _forward(self, batch_tensor: torch.Tensor):
        """Run inference and return raw outputs in backend-native format."""
        if self.backend == "pytorch":
            # FP32 forward under ``torch.no_grad()``; no autocast.

            with torch.no_grad():
                return self.model.model(batch_tensor)

        if not self.backend.startswith("onnx"):
            raise RuntimeError(f"Invalid backend: {self.backend}")

        if self.device == "cpu":
            # CPU: numpy input via session.run
            inp = batch_tensor.detach().cpu().numpy()
            inp = np.ascontiguousarray(inp, dtype=np.float32)

            return self.session.run(self.output_names, {self.input_name: inp})

        # CUDA + IO Binding via DLPack
        io_binding = self.session.io_binding()

        batch_tensor = batch_tensor.contiguous()

        ort_input = ort.OrtValue.from_dlpack(to_dlpack(batch_tensor))

        io_binding.bind_ortvalue_input(self.input_name, ort_input)

        for out in self.output_names:
            io_binding.bind_output(out, "cuda")

        # The input tensor is produced asynchronously on the torch stream
        # (``_assemble_batch``: pinned H2D copy + in-place normalization),
        # while ORT executes on its own stream. The bound pointer is read
        # without a cross-stream barrier, so the read can race the producing
        # kernels — see docs/ARCHITECTURE.md §11, CUDA finding 3.
        # Synchronize the producing stream before handing the buffer to ORT.
        torch.cuda.current_stream().synchronize()

        self.session.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()

    # run one preprocessed batch through forward + post_process
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

        Used by both ``infer`` (file-backed) and ``infer_frames``
        (array-backed).
        """
        outputs = self._forward(data["images"])
        logger.debug("Backend=%s output type=%s shape=%s dtype=%s",
                     self.backend, type(outputs), getattr(outputs, "shape", None),
                     getattr(outputs, "dtype", None))
        # Guard: expected output types are np.ndarray, torch.Tensor, list/tuple.
        if not isinstance(outputs, (np.ndarray, torch.Tensor, list, tuple)):
            raise TypeError(
                f"unexpected output type {type(outputs)} for backend {self.backend}"
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
            # ``nc=len(self.class_names)`` for exact output-layout resolution.
            nc=len(self.class_names) if self.class_names else None,
        )

        if save:
            for j, dets in enumerate(batch_dets):
                if not dets:
                    continue
                # Save path includes the source parent directory name to avoid
                # filename collisions.
                src_path = Path(data["paths"][j])
                parent_tag = src_path.parent.name
                stem = f"{parent_tag}_{src_path.name}" if parent_tag else src_path.name
                save_path = output_dir / f"result_{stem}"
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

    # infer
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
        """Run end-to-end inference: preprocess -> forward -> postprocess.

        Returns a list of ``[(x1,y1,x2,y2,conf,cls), ...]`` per image, in original-image
        coordinates.
        """
        image_paths = load_images(imgs_input, max_images=max_imgs)
        output_dir = Path(output_dir)
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)

        all_results: List[List[tuple]] = []
        failed_batches: List[tuple] = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            try:
                data = preprocess_imgs(
                    batch_paths,
                    imgsz=self.imgsz,
                    device=self.device,
                    original=save,
                )
                batch_dets = self._run_batch(
                    data, conf=conf, iou=iou, max_det=max_det,
                    save=save, output_dir=output_dir,
                )
                all_results.extend(batch_dets)
                logger.info(
                    "Batch %d | Detections: %d",
                    i // batch_size + 1,
                    sum(len(x) for x in batch_dets),
                )
            except Exception as e:
                failed_batches.append((i // batch_size + 1, batch_paths, e))
                logger.exception("Batch inference failed (paths=%s)", batch_paths)
        if failed_batches:
            logger.warning(
                "%d/%d batches failed and were skipped from the results: %s",
                len(failed_batches), (len(image_paths) + batch_size - 1) // batch_size,
                [msg for _, _, msg in failed_batches],
            )
        return all_results

    # infer from in-memory BGR frames (no file I/O) — server hot path
    def infer_frames(
        self,
        frames: List[np.ndarray],
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        save: bool = False,
        output_dir: Union[str, Path] = "results/predictions",
    ) -> List[List[tuple]]:
        """Inference from already-decoded BGR frames (no file I/O).

        Frames are preprocessed via ``preprocess_frames`` and run through the
        same ``_run_batch`` pipeline as the CLI's ``infer``.
        """
        output_dir = Path(output_dir)
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)
        data = preprocess_frames(
            frames,
            imgsz=self.imgsz,
            device=self.device,
            original=save,
        )
        return self._run_batch(
            data, conf=conf, iou=iou, max_det=max_det,
            save=save, output_dir=output_dir,
        )
