"""Letterbox preprocessing and batch tensor assembly for YOLOv8 inference.

The pipeline is:

    Raw Image (HWC BGR) -> Letterbox -> BGR->RGB -> HWC->CHW -> /255 -> float32 NCHW

Both single-image (``preprocess_single``) and batched (``preprocess_imgs``) APIs are exposed. The
batched variant uses a ``ThreadPoolExecutor`` for the per-image cv2 work; the rest of the pipeline
(numpy stack, ``torch.from_numpy``, normalization) is single- threaded and runs once per batch.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from utils import get_logger
from utils.threading import clamp_workers

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------
def check_imgsz(imgsz: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Validate and normalize target image size to a ``(height, width)`` tuple.

    The (h, w) convention matches Ultralytics' ``letterbox`` (``shape`` is ``(h, w)`` and ``r =
    min(new_shape[0]/shape[0], new_shape[1]/shape[1])``), which the letterbox math here mirrors. For
    square inputs (the default ``640``) the ordering is irrelevant; it only matters for non-square
    tuples.
    """
    if isinstance(imgsz, int):
        return (imgsz, imgsz)
    if isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
        return tuple(imgsz)
    raise ValueError(f"Invalid imgsz: {imgsz}")


# ---------------------------------------------------------------------------
# Letterbox
# ---------------------------------------------------------------------------
def letterbox(
    img: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """Resize and pad an image while preserving aspect ratio.

    Returns
    -------
    img : np.ndarray
        Padded image, shape ``(new_h, new_w, 3)``.
    ratio : (float, float)
        ``(r_w, r_h)`` — scale factors applied to width and height.
    pad : (int, int)
        ``(dw, dh)`` — *total* padding along width and height. Single-side padding (for
        ``scale_boxes``) is ``(dw/2, dh/2)``.
    """
    shape = img.shape[:2]  # (h, w)
    new_shape = check_imgsz(new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = (r, r)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw %= stride
        dh %= stride
    elif scale_fill:
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    left, top = int(dw / 2), int(dh / 2)
    right, bottom = dw - left, dh - top

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


# ---------------------------------------------------------------------------
# Single image
# ---------------------------------------------------------------------------
def preprocess_single(
    img_path: Union[str, Path],
    imgsz: Union[int, Tuple[int, int]] = 640,
    original: bool = False,
) -> Optional[dict]:
    """Preprocess a single image. Returns ``None`` on failure (logged)."""
    img_path = Path(img_path)
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        # Grayscale / BGRA -> BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        orig = img.copy() if original else None
        orig_shape = img.shape[:2]

        img, ratio, pad = letterbox(img, new_shape=imgsz)
        pad_left = int(pad[0] / 2)
        pad_top = int(pad[1] / 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))

        return {
            "img": img,
            "orig_img": orig,
            "orig_shape": orig_shape,
            "ratio": ratio,                # (r_w, r_h)
            "pad": (pad_left, pad_top),    # single-side padding
            "path": str(img_path),
        }
    except Exception as e:
        logger.error("Preprocessing failed %s: %s", img_path, e, exc_info=False)
        return None


# ---------------------------------------------------------------------------
# Batched preprocessing
# ---------------------------------------------------------------------------
def preprocess_imgs(
    img_paths: List[Union[str, Path]],
    imgsz: Union[int, Tuple[int, int]] = 640,
    device: str = "cpu",
    fp16: bool = False,
    original: bool = False,
    num_workers: int = 4,
    disable_cv2_threading: bool = True,
) -> Dict:
    """Batch-preprocess images for YOLOv8 inference or INT8 calibration.

    Parameters
    ----------
    disable_cv2_threading
        If True, call ``cv2.setNumThreads(0)`` once before launching the worker pool. This avoids
        oversubscribing CPUs — OpenCV otherwise spawns its own thread pool inside each worker. The
        setting is global and sticky for the process; safe to call repeatedly.

    Returns
    -------
    dict with keys ``images``, ``orig_imgs``, ``orig_shapes``, ``ratios``, ``pads``, ``paths``.
    """
    if not img_paths:
        raise ValueError("Image path list cannot be empty")

    if disable_cv2_threading:
        # Idempotent — OpenCV itself guards against redundant calls.
        cv2.setNumThreads(0)

    img_paths = [Path(p) for p in img_paths]
    imgsz_t = check_imgsz(imgsz)

    # Cap worker count at 2×CPU: extra threads oversubscribe the kernel
    # scheduler and *reduce* throughput for the cv2-letterbox work this pool
    # actually runs. ``clamp_workers`` warns when it kicks in so the caller
    # sees that the requested value was bound.
    workers = clamp_workers(num_workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(
            lambda p: preprocess_single(p, imgsz=imgsz_t, original=original),
            img_paths,
        ))

    valid = [r for r in results if r is not None]
    if not valid:
        raise RuntimeError("All images failed preprocessing")
    if len(valid) < len(img_paths):
        logger.warning(
            "Some images failed preprocessing: %d/%d skipped",
            len(img_paths) - len(valid), len(img_paths),
        )

    # Assemble batch
    imgs = [r["img"] for r in valid]
    orig_imgs = [r["orig_img"] for r in valid] if original else []
    orig_shapes = [r["orig_shape"] for r in valid]
    ratios = [r["ratio"] for r in valid]
    pads = [r["pad"] for r in valid]
    paths = [r["path"] for r in valid]

    batch_np = np.stack(imgs, axis=0)  # (B, 3, H, W) uint8
    tensor = torch.from_numpy(batch_np).to(torch.float32)

    actual_device = device
    if device == "cuda" and torch.cuda.is_available():
        # pin_memory only meaningful for host tensors being copied to CUDA
        tensor = tensor.pin_memory().to(device, non_blocking=True)
    else:
        actual_device = "cpu"

    tensor /= 255.0  # normalize to [0, 1]

    if fp16:
        if actual_device == "cuda":
            tensor = tensor.half()
        else:
            logger.warning("FP16 is ineffective on CPU; ignoring")

    logger.debug(
        "Preprocessing done | count=%d | shape=%s | dtype=%s | device=%s",
        len(valid), tuple(tensor.shape), tensor.dtype, actual_device,
    )

    return {
        "images": tensor,
        "orig_imgs": orig_imgs,
        "orig_shapes": orig_shapes,
        "ratios": ratios,
        "pads": pads,
        "paths": paths,
    }
