"""Detection visualization helpers.

Color palette is keyed on **class id** (stable across frames and runs),
giving every class a consistent color and making annotated images easier to
read at a glance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color helpers — stable per class id
# ---------------------------------------------------------------------------
# High-saturation BGR palette; the 12 entries cover the COCO 12-class subset
# and further ids cycle. Label text color is picked per background luminance,
# so bright entries (yellow, lime, cyan) get black text and the rest white.
_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (75, 25, 230),    # red
    (75, 180, 60),    # green
    (25, 225, 255),   # yellow
    (200, 130, 0),    # blue
    (48, 130, 245),   # orange
    (180, 30, 145),   # purple
    (240, 240, 70),   # cyan
    (230, 50, 240),   # magenta
    (60, 245, 210),   # lime
    (128, 128, 0),    # teal
    (0, 0, 128),      # maroon
    (128, 0, 0),      # navy
)

# Rec.601 luma threshold separating dark label backgrounds (white text) from
# bright ones (black text).
_LUMA_TEXT_CUTOFF = 160.0


def color_for_class(cls_id: int) -> Tuple[int, int, int]:
    """Return the class's BGR color from the palette (stable, cycles)."""
    return _PALETTE[cls_id % len(_PALETTE)]


def _text_color(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """White on dark label backgrounds, black on bright ones."""
    b, g, r = bg
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luma > _LUMA_TEXT_CUTOFF else (255, 255, 255)


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

    Each detection is ``(x1, y1, x2, y2, conf, cls)``. If ``conf_thres`` is set,
    detections below it are dropped — useful when callers want to draw with a
    stricter cutoff than inference used.
    """
    if not detections:
        return img

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf_thres is not None and conf < conf_thres:
            continue

        color = color_for_class(int(cls))
        txt_color = _text_color(color)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, thickness, lineType=cv2.LINE_AA)

        label = (
            f"{class_names[int(cls)] if class_names else int(cls)} "
            f"{conf:.2f}"
        )
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )
        xi, yi = int(x1), int(y1)
        label_h = th + 5
        # Labels sit above the box; when the box is flush with the top edge,
        # the label is drawn inside it.
        if yi - label_h >= 0:
            top, baseline_y = yi - label_h, yi - 4
        else:
            top, baseline_y = yi, yi + th + 1
        cv2.rectangle(img, (xi, top), (xi + tw, top + label_h), color, -1)
        # 1px border in the text color separates the label from background
        # pixels of a similar color.
        cv2.rectangle(img, (xi, top), (xi + tw, top + label_h), txt_color, 1)
        cv2.putText(
            img, label,
            (xi, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, txt_color, 2, cv2.LINE_AA,
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
