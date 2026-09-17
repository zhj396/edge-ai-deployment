"""YOLOv8 post-processing — NMS + scale-back to original image coordinates.

The pipeline:

    raw output (bs, 4+nc, num_boxes)  --NMS-->  boxes (bs, N, 6)
                                                     |
                                                     +-- scale_boxes  (letterbox -> orig)
                                                     +-- clip         (per-image bounds)
                                                     +-- format       (tuple per detection)

Both ``torch.Tensor`` and ``np.ndarray`` outputs are supported, and the input may be either (bs, N,
4+nc) or (bs, 4+nc, N). NMS itself is delegated to Ultralytics so behavior matches the official
YOLOv8.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

from utils import get_logger

logger = get_logger(__name__)


def _ensure_4nc_first(
    pred: torch.Tensor, nc: Optional[int] = None
) -> torch.Tensor:
    """Permute ``pred`` so the class channel dim is at axis 1.

    Two layouts are common:

    * ``(bs, 4+nc, N)`` — direct ORT output for YOLOv8 export. No permute needed.
    * ``(bs, N, 4+nc)`` — direct PyTorch forward (Ultralytics' raw head).

    For YOLOv8s at 640px, ``N=8400 >> 4+nc=16``, so the smaller trailing dim is reliably ``4+nc``
    and a magnitude heuristic decides correctly. The ``SMALL=32`` threshold guards against tiny
    *test* tensors where the magnitude rule alone is unsafe — e.g. a channels-first
    ``(bs, 4+nc=8, N=7)`` tensor satisfies ``shape[1] > shape[2]`` but is already laid out
    correctly and must not be permuted.

    Passing ``nc`` opts out of the heuristic and resolves the layout exactly: whichever dim equals
    ``4+nc`` is the channel dim; the other is ``N``. Use this when ``num_boxes`` genuinely drops
    below ``4+nc`` (a heavily pruned detector or a tiny custom head) and the magnitude rule is
    unsafe.
    """
    SMALL = 32  # YOLOv8s @ 640: N=8400 >> 4+nc=16; below this we can't trust magnitudes.
    if nc is not None:
        # Exact: the dim equal to 4+nc is the channel dim.
        if pred.shape[1] == 4 + nc:
            return pred.contiguous()  # already (bs, 4+nc, N)
        if pred.shape[2] == 4 + nc:
            return pred.permute(0, 2, 1).contiguous()  # (bs, N, 4+nc) → (bs, 4+nc, N)
        raise ValueError(
            f"Cannot resolve pred layout {tuple(pred.shape)} to (bs, {4 + nc}, N); "
            f"neither dim matches 4+nc. Pass an explicit nc, transpose manually, "
            f"or check that the model output is (bs, *, 4+nc) / (bs, 4+nc, *)."
        )
    # Magnitude heuristic for the standard YOLOv8 case.
    if pred.shape[1] > pred.shape[2] and pred.shape[1] > SMALL:
        return pred.permute(0, 2, 1).contiguous()
    return pred


def post_process(
    outputs: Union[np.ndarray, torch.Tensor, List, Tuple],
    orig_shapes: List[Tuple[int, int]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    imgsz: Union[int, Tuple[int, int]] = 640,
    max_det: int = 300,
    ratios: Optional[List[Tuple[float, float]]] = None,
    pads: Optional[List[Tuple[float, float]]] = None,
    agnostic_nms: bool = False,
    nc: Optional[int] = None,
) -> List[List[Tuple[int, int, int, int, float, int]]]:
    """Post-process raw model output to final detections.

    Parameters
    ----------
    nc
        Number of classes (e.g. ``12`` for the COCO-subset project). When supplied, ``post_process``
        resolves the prediction layout exactly (whichever dim equals ``4+nc`` is the channel
        dim) — safe for tiny or pruned detectors where ``num_boxes < 4+nc``. When ``None``
        (default), a magnitude heuristic is used (correct for YOLOv8 at standard sizes).

    Returns
    -------
    ``[[(x1, y1, x2, y2, conf, cls), ...], ...]`` — one inner list per image, in
    **original** image coordinates (clipped to image bounds, rounded to int pixels,
    scores rounded to 4 decimals).
    """
    try:
        # ---- 1. extract prediction tensor ----
        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 0:
                raise ValueError("outputs is empty")
            pred = outputs[0]
        else:
            pred = outputs

        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)

        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"Unsupported prediction type: {type(pred)}")
        if pred.ndim != 3:
            raise ValueError(f"Wrong prediction dimensions: {pred.ndim}D, expected 3D")

        pred = _ensure_4nc_first(pred, nc=nc)
        pred = pred.float()  # float32 (INT8 quantize output can be lower precision)

        bs = pred.shape[0]
        if len(orig_shapes) != bs:
            raise ValueError(
                f"orig_shapes count mismatch: {len(orig_shapes)} != {bs}"
            )
        if ratios is not None and len(ratios) != bs:
            raise ValueError(f"ratios count mismatch: {len(ratios)} != {bs}")
        if pads is not None and len(pads) != bs:
            raise ValueError(f"pads count mismatch: {len(pads)} != {bs}")

        img1_shape = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz

        # ---- 2. NMS ----
        results = non_max_suppression(
            prediction=pred,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            agnostic=agnostic_nms,
        )

        # ---- 3. per-image: scale + clip + format ----
        batch_detections: List[List[Tuple[int, int, int, int, float, int]]] = []

        for i, det in enumerate(results):
            if len(det) == 0:
                batch_detections.append([])
                continue

            det = det.cpu()
            ratio_pad = (ratios[i], pads[i]) if (ratios and pads) else None

            scaled = scale_boxes(
                img1_shape=img1_shape,
                boxes=det[:, :4],
                img0_shape=orig_shapes[i],
                ratio_pad=ratio_pad,
            )

            # Vectorized clip + format.  ``scaled`` is (M, 4) torch tensor.
            h, w = orig_shapes[i]
            scaled[:, 0] = scaled[:, 0].clamp(0, w - 1)
            scaled[:, 1] = scaled[:, 1].clamp(0, h - 1)
            scaled[:, 2] = scaled[:, 2].clamp(0, w - 1)
            scaled[:, 3] = scaled[:, 3].clamp(0, h - 1)
            scaled = scaled.round().int()

            confs = det[:, 4]
            clses = det[:, 5].int()

            detections: List[Tuple[int, int, int, int, float, int]] = []
            for j in range(scaled.shape[0]):
                x1, y1, x2, y2 = scaled[j].tolist()
                detections.append(
                    (int(x1), int(y1), int(x2), int(y2),
                     round(float(confs[j]), 4), int(clses[j]))
                )
            batch_detections.append(detections)

        logger.debug(
            "Post-processing done | batch=%d | total_boxes=%d",
            bs, sum(len(d) for d in batch_detections),
        )
        return batch_detections

    except Exception as e:
        logger.error("Post-processing failed: %s", e, exc_info=True)
        raise
