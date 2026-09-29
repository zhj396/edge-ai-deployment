"""Backend-agnostic COCO mAP evaluator (pure NumPy + a thin engine driver).

Lives in ``utils`` (not ``src``) so it stays importable without torch /
ultralytics / onnxruntime — the IoU + AP + label-parsing functions are pure
NumPy/Python and are unit-tested with hand-computed synthetic scenarios
(model-free, matching the project's test posture, CLAUDE.md invariant 7). The
driver :func:`evaluate_map` receives a duck-typed ``engine`` exposing the
shared ``infer()`` API (``src/engine.py``, ``src/openvino_engine.py`` — both
return ``List[List[(x1,y1,x2,y2,conf,cls)]]`` in original-image pixel coords)
and never imports ultralytics itself.

Scope: Ultralytics' ``YOLO(path).val()`` drives only artifacts carrying
ultralytics metadata (``.pt`` or its own exports); this evaluator accepts any
engine exposing the shared ``infer()`` API, raw OpenVINO IRs included, and
applies one COCO metric to all of them. See docs/ARCHITECTURE.md §8.1 for
the benchmark-side mAP-provenance note.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Must match utils/common.py::load_images valid_suffix so the val-image sort
# here and the engine's internal load_images sort produce the SAME order
# (load_images resolves and sorts the image set — see
# utils/common.py::load_images). A mismatch silently maps detections to
# the wrong image stems → wrong mAP.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# COCO mAP IoU thresholds: mAP@.5 and mAP@.5:.95 (mean over 0.5..0.95 step .05).
IOU_THRESHOLDS = np.round(np.arange(0.5, 1.0, 0.05), 2)  # 0.5,0.55,...,0.95
RECALLinterp = np.linspace(0.0, 1.0, 101)  # COCO 101-point interpolation


# ---------------------------------------------------------------------------
# Vectorized IoU
# ---------------------------------------------------------------------------
def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU matrix for axis-aligned xyxy boxes. Returns ``(Na, Nb)``.

    ``boxes_a`` is ``(Na, 4)``, ``boxes_b`` is ``(Nb, 4)`` (xyxy). Empty either
    side → ``np.zeros((Na, Nb))`` (never NaN; a degenerate box has area 0 and
    IoU 0). Pure NumPy, vectorized via broadcast.
    """
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 1, 4)  # (Na,1,4)
    b = np.asarray(boxes_b, dtype=float).reshape(1, -1, 4)  # (1,Nb,4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[1]))
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


# ---------------------------------------------------------------------------
# Ground-truth loader
# ---------------------------------------------------------------------------
def _resolve_label_dir(image_dir: Path) -> Path:
    """images/val -> labels/val (segment swap, not substring replace).

    Mirrors ``src/sampler.py:95-111``: on Windows ``str(Path)`` uses
    backslashes, so ``replace("/images/", "/labels/")`` silently no-ops. A
    path-segment swap is cross-platform. Falls back to a sibling
    ``<parent>/labels/<name>`` when there is no ``images`` segment.
    """
    parts = list(image_dir.parts)
    if "images" in parts:
        i = parts.index("images")
        parts[i] = "labels"
        return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def load_ground_truth(
    data_yaml: Path,
) -> Tuple[List[Path], Dict[str, np.ndarray], Dict[int, str]]:
    """Read the YOLO val set + labels from ``data.yaml``.

    Returns ``(sorted resolved image paths, {stem: (N,5) [cls,x1,y1,x2,y2]
    pixel xyxy}, {cls_id: name})``. An image with no / empty label file gets a
    ``(0,5)`` array — it is kept (its detections become FP; it contributes to
    no class's GT count). Class names come from the ``names`` field (list or
    dict), via the pure-Python ``_class_names_from_data_yaml`` helper.
    """
    import yaml  # lazy: keeps the module importable without pyyaml at top

    data_yaml = Path(data_yaml)
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names")
    if isinstance(names, dict):
        class_names = {int(k): v for k, v in names.items()}
    elif isinstance(names, list):
        class_names = {i: n for i, n in enumerate(names)}
    else:
        raise ValueError(f"data.yaml {data_yaml} has no 'names' field")

    image_dir = (data_yaml.parent / cfg["val"]).resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Val image dir not found: {image_dir}")
    label_dir = _resolve_label_dir(image_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {label_dir}")

    # sorted(resolved) so engine.infer's internal sorted(set(resolved)) is a
    # no-op → results[i] aligns with paths[i] (see module IMAGE_SUFFIXES note).
    image_paths = sorted(
        (p.resolve() for p in image_dir.iterdir()
         if p.suffix.lower() in IMAGE_SUFFIXES),
    )

    try:
        from PIL import Image  # lazy: PIL is a dep but keep top clean
    except ImportError as e:  # pragma: no cover — Pillow is a base dep
        raise ImportError(
            "Pillow is required to read image dims for GT box scaling: "
            f"{e}. pip install pillow"
        ) from e

    gts: Dict[str, np.ndarray] = {}
    seen_stems = set()
    for img_path in image_paths:
        # Detections and GT are keyed by stem; two val images sharing a
        # stem (img.jpg + img.png) would silently overwrite each other's
        # entry, so the collision is rejected up front.
        if img_path.stem in seen_stems:
            raise ValueError(
                f"Duplicate image stem {img_path.stem!r} in {image_dir} "
                f"({img_path.name} shares its stem with another val "
                "image); mAP keying is per stem."
            )
        seen_stems.add(img_path.stem)
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            gts[img_path.stem] = np.zeros((0, 5), dtype=float)
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        rows = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                cls = int(float(p[0]))
                cx, cy, bw, bh = map(float, p[1:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                rows.append([cls, x1, y1, x2, y2])
        gts[img_path.stem] = (
            np.array(rows, dtype=float) if rows else np.zeros((0, 5), dtype=float)
        )
    return image_paths, gts, class_names


# ---------------------------------------------------------------------------
# COCO AP
# ---------------------------------------------------------------------------
def _ap_per_class(
    dets_c: List[Tuple[float, str, np.ndarray]],  # (conf, stem, box(4,))
    gts: Dict[str, np.ndarray],
    c: int,
    iou_threshold: float,
) -> Tuple[float, int]:
    """AP for one class at one IoU threshold, plus the class's GT count.

    Greedy conf-descending match, pycocotools' own matcher semantics: each
    detection picks the highest-IoU same-image **unmatched** GT of class
    ``c`` with IoU >= threshold (TP; already-matched GTs are skipped and the
    search continues among the rest), else FP. One GT per detection, one
    detection per GT. Crowd/ignore regions are not modeled (YOLO labels
    carry no iscrowd). Differences from pycocotools' *summary* metric on
    the same detections: maxDets is the engine's per-image cap (300 by
    default, vs COCO's official 100), and confidence ties are broken
    deterministically by (image, box position). Keep these in mind when
    comparing against Ultralytics/pycocotools numbers.
    Returns ``(ap, num_gt_c)``; ``ap=0.0`` if
    there are no GT (the caller excludes zero-GT classes from the mean).
    """
    # GT boxes of class c per image + a per-image "matched" flag array.
    gt_by_img: Dict[str, np.ndarray] = {}
    matched_by_img: Dict[str, np.ndarray] = {}
    num_gt = 0
    for stem, arr in gts.items():
        mask = arr[:, 0].astype(int) == c if arr.shape[0] else np.zeros(0, bool)
        boxes = arr[mask, 1:5] if arr.shape[0] else np.zeros((0, 4))
        gt_by_img[stem] = boxes
        matched_by_img[stem] = np.zeros(boxes.shape[0], dtype=bool)
        num_gt += boxes.shape[0]

    if num_gt == 0:
        return 0.0, 0

    # Sort dets by conf DESC, tiebreak (stem, x1, y1) ASC for determinism.
    dets_c.sort(key=lambda d: (-d[0], d[1], float(d[2][0]), float(d[2][1])))

    tp = np.zeros(len(dets_c), dtype=float)
    fp = np.zeros(len(dets_c), dtype=float)
    for i, (_conf, stem, box) in enumerate(dets_c):
        cand = gt_by_img.get(stem)
        if cand is None or cand.shape[0] == 0:
            fp[i] = 1.0
            continue
        ious = box_iou(box.reshape(1, 4), cand)[0]  # (num_gt_in_img,)
        unmatched = ~matched_by_img[stem]
        # Zero the IoU of already-matched GTs so they can't be re-chosen.
        ious = np.where(unmatched, ious, -1.0)
        j = int(np.argmax(ious))
        if ious[j] >= iou_threshold:
            tp[i] = 1.0
            matched_by_img[stem][j] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / num_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # No detections at all → recall 0 at every interp point → AP 0. Guard
    # explicitly: the 101-point indexing below requires a non-empty
    # precision array.
    if precision.shape[0] == 0:
        return 0.0, num_gt

    # COCO 101-point interpolation: for each r, the max precision among PR
    # points whose recall >= r (the monotonic envelope). A point at recall R
    # covers all interpolation r in [0, R].
    # Make precision monotonically decreasing from the right (envelope) so
    # "max precision where recall >= r" is a right-to-left cumulative max.
    for k in range(len(precision) - 1, 0, -1):
        precision[k - 1] = max(precision[k - 1], precision[k])
    # precision is now non-increasing; recall is non-decreasing.
    # For each r, find the first PR point with recall >= r; its precision is
    # the envelope value (points to its right have >= recall and <= precision).
    idx = np.searchsorted(recall, RECALLinterp, side="left")
    prec_interp = np.where(idx < len(precision), precision[np.clip(idx, 0, len(precision) - 1)], 0.0)
    ap = float(np.mean(prec_interp))
    return ap, num_gt


def compute_map(
    detections: Dict[str, List[tuple]],
    gts: Dict[str, np.ndarray],
    num_classes: int,
) -> Tuple[Optional[float], Optional[float], List[Dict]]:
    """COCO mAP50 / mAP50-95 + per-class rows.

    ``detections``: ``{stem: [(cls, x1,y1,x2,y2, conf), ...]}``.
    ``gts``: ``{stem: (N,5) [cls,x1,y1,x2,y2]}``.
    Returns ``(mAP50, mAP50-95, [{class, mAP50, mAP50-95}, ...])``. Classes with
    zero GT in the whole val set are EXCLUDED from the mean (COCO behavior —
    not counted as 0) and get ``None`` values in their per-class row.
    """
    # Bucket detections by class across all images.
    dets_by_class: Dict[int, List[Tuple[float, str, np.ndarray]]] = {
        c: [] for c in range(num_classes)
    }
    for stem, dl in detections.items():
        for det in dl:
            cls = int(det[0])
            conf = float(det[5])
            box = np.array(det[1:5], dtype=float)
            if 0 <= cls < num_classes:
                dets_by_class[cls].append((conf, stem, box))

    per_class_rows: List[Dict] = []
    ap50_list: List[float] = []
    ap5095_list: List[float] = []
    for c in range(num_classes):
        dets_c = dets_by_class[c]
        aps = []
        num_gt_c = 0
        for t in IOU_THRESHOLDS:
            ap, ng = _ap_per_class(dets_c, gts, c, float(t))
            aps.append(ap)
            num_gt_c = ng  # same across thresholds
        if num_gt_c == 0:
            # No GT for this class in the val set — exclude from the mean.
            per_class_rows.append({"class": c, "mAP50": None, "mAP50-95": None})
            continue
        ap50 = float(aps[0])          # threshold 0.5
        ap5095 = float(np.mean(aps))  # mean over 0.5..0.95
        ap50_list.append(ap50)
        ap5095_list.append(ap5095)
        per_class_rows.append({"class": c, "mAP50": ap50, "mAP50-95": ap5095})

    map50 = float(np.mean(ap50_list)) if ap50_list else None
    map5095 = float(np.mean(ap5095_list)) if ap5095_list else None
    return map50, map5095, per_class_rows


# ---------------------------------------------------------------------------
# Driver — runs an engine's infer() over the val set and computes mAP
# ---------------------------------------------------------------------------
def evaluate_map(
    engine,
    data_yaml: Path,
    conf: float,
    iou: float,
    batch_size: int,
    max_det: int,
    backend_name: str,
    timestamp_suffix: str,
    class_names_override: Optional[dict] = None,
) -> Optional[Dict]:
    """Drive ``engine.infer(...)`` over the val set, compute + dump mAP.

    The engine must expose ``infer(imgs_input, conf, iou, max_imgs, batch_size,
    max_det, save, output_dir) -> List[List[(x1,y1,x2,y2,conf,cls)]]`` (the
    shared API of YOLOv8Engine / OpenVINOEngine). The engine carries its own
    imgsz — it must be constructed with the size its artifact serves.
    conf=0.001, iou=0.7 are the COCO mAP-standard NMS config (the benchmark's
    ``conf_threshold`` / ``iou_threshold``). Writes the per-class CSV
    (``results/<backend>_perclass<TS>.csv``, 4-col schema). Returns
    ``{"mAP50", "mAP50-95"}`` (values None if every class has zero GT), or
    ``None`` if the run crashed.
    """
    try:
        import pandas as pd  # lazy: src/benchmark.py already deps pandas
    except ImportError as e:  # pragma: no cover
        raise ImportError(f"pandas required to write per-class CSV: {e}") from e

    tmpdir = tempfile.mkdtemp(prefix="map_eval_")
    try:
        val_paths, gts, yaml_names = load_ground_truth(Path(data_yaml))
        names = class_names_override if class_names_override else yaml_names
        if not names:
            logger.warning("%s: no class names resolved; aborting mAP", backend_name)
            return None
        num_classes = max(int(k) for k in names) + 1 if names else 0

        # max_imgs MUST be len(val_paths): infer's default 32 would silently
        # truncate the val set → wrong mAP. save=False skips annotated-image
        # writes; output_dir is required by infer's signature regardless.
        results = engine.infer(
            imgs_input=val_paths,
            conf=conf,
            iou=iou,
            max_imgs=len(val_paths),
            batch_size=batch_size,
            max_det=max_det,
            save=False,
            output_dir=tmpdir,
        )
        if results is None or len(results) != len(val_paths):
            logger.error(
                "%s: infer returned %d results for %d val images (order/count "
                "mismatch); aborting mAP",
                backend_name, -1 if results is None else len(results),
                len(val_paths),
            )
            return None

        detections: Dict[str, List[tuple]] = {}
        for i, dets in enumerate(results):
            stem = val_paths[i].stem
            detections[stem] = [
                (int(d[5]), float(d[0]), float(d[1]), float(d[2]),
                 float(d[3]), float(d[4]))
                for d in (dets or [])
            ]

        map50, map5095, rows = compute_map(detections, gts, num_classes)

        # Per-class CSV: 4-col schema (backend, class, mAP50, mAP50-95),
        # matching the summary CSV's mAP columns.
        csv_rows = []
        for r in rows:
            name = names.get(r["class"], f"class_{r['class']}")
            csv_rows.append([
                backend_name, name,
                r["mAP50"] if r["mAP50"] is not None else "",
                r["mAP50-95"] if r["mAP50-95"] is not None else "",
            ])
            logger.info(
                "%s | %s: mAP50=%s, mAP50-95=%s", backend_name, name,
                f"{r['mAP50']:.4f}" if r['mAP50'] is not None else "n/a (no GT)",
                f"{r['mAP50-95']:.4f}" if r['mAP50-95'] is not None else "n/a",
            )
        logger.info(
            "%s Overall | mAP50=%s, mAP50-95=%s", backend_name,
            f"{map50:.4f}" if map50 is not None else "n/a",
            f"{map5095:.4f}" if map5095 is not None else "n/a",
        )

        os.makedirs("results", exist_ok=True)
        csv_path = f"results/{backend_name}_perclass{timestamp_suffix}.csv"
        pd.DataFrame(csv_rows, columns=["backend", "class", "mAP50", "mAP50-95"]) \
            .to_csv(csv_path, index=False)
        logger.info("%s per-class metrics saved: %s", backend_name, csv_path)

        return {"mAP50": map50, "mAP50-95": map5095}
    except Exception as e:
        logger.exception("%s native mAP evaluation failed: %s", backend_name, e)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
