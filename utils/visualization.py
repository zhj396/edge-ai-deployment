"""Detection visualization helpers.

Color palette is keyed on **class id** (stable across frames) rather than detection index — this
gives every class a consistent color and makes annotated images easier to read at a glance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color helpers — stable per class id
# ---------------------------------------------------------------------------
_PALETTE_CACHE: Dict[int, Tuple[int, int, int]] = {}


def color_for_class(cls_id: int) -> Tuple[int, int, int]:
    """Return a deterministic BGR color for a class id."""
    if cls_id not in _PALETTE_CACHE:
        # md5 of the class id -> stable across runs without persistence
        h = hashlib.md5(str(cls_id).encode("utf-8")).digest()
        # Bright colors only — keep max channel >= 128 for visibility
        r, g, b = h[0], h[1], h[2]
        # Re-bias to avoid dim colors on dark backgrounds
        r = 128 + (r % 128)
        g = 128 + (g % 128)
        b = 128 + (b % 128)
        _PALETTE_CACHE[cls_id] = (int(b), int(g), int(r))  # OpenCV uses BGR
    return _PALETTE_CACHE[cls_id]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_detections(
    img: np.ndarray,
    detections: List[Tuple],
    class_names: Optional[Dict[int, str]] = None,
    conf_thres: Optional[float] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw bounding boxes and class labels on the image.

    Each detection is ``(x1, y1, x2, y2, conf, cls)``. If ``conf_thres`` is set, detections below it
    are dropped — useful when callers want to draw with a stricter cutoff than inference used.
    """
    if not detections:
        return img

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf_thres is not None and conf < conf_thres:
            continue

        color = color_for_class(int(cls))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, thickness)

        label = (
            f"{class_names[int(cls)] if class_names else int(cls)} "
            f"{conf:.2f}"
        )
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )
        # Background box
        cv2.rectangle(
            img,
            (int(x1), int(y1) - th - 5),
            (int(x1) + tw, int(y1)),
            color, -1,
        )
        cv2.putText(
            img, label,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return img


def save_annotated_image(
    orig_img: np.ndarray,
    detections: List[Tuple],
    save_path: str,
    class_names: Optional[Dict[int, str]] = None,
    conf_thres: Optional[float] = None,
) -> np.ndarray:
    """Draw detections and save to disk. Returns the annotated image."""
    annotated = draw_detections(
        orig_img.copy(),
        detections,
        class_names=class_names,
        conf_thres=conf_thres,
    )
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_p), annotated)
    return annotated
