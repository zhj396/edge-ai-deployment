"""Unit tests for src/postprocess.py — pure-tensor logic, no model needed."""
import pytest
import torch

from src.postprocess import post_process


# YOLOv8-style prediction layout: (bs, 4+nc, num_boxes).
# We use ``num_boxes >= 8`` so the layout heuristic (shape[1] vs shape[2])
# never confuses (bs, 4+nc, N) with (bs, N, 4+nc).

NC = 3


def _make_pred(boxes_xywh_norm, cls_scores, nc=NC):
    """Build a (bs=1, 4+nc, N) tensor from XYWH-normalized boxes + class scores.

    Parameters
    ----------
    boxes_xywh_norm : list of [cx, cy, w, h]
    cls_scores : list of length-nc lists — class scores *per box*
    """
    boxes = torch.tensor(boxes_xywh_norm, dtype=torch.float32)
    cls = torch.tensor(cls_scores, dtype=torch.float32)
    n = boxes.shape[0]
    assert cls.shape == (n, nc)
    pred = torch.cat([boxes, cls], dim=1)             # (N, 4 + nc)
    return pred.unsqueeze(0).permute(0, 2, 1).contiguous()  # (1, 4 + nc, N)


def test_postprocess_tensor_input_runs():
    # Two non-overlapping boxes, both with conf > 0.5
    boxes = [[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.2, 0.2]]
    cls = [[0.9, 0.0, 0.0], [0.8, 0.0, 0.0]]
    pred = _make_pred(boxes, cls)
    # Pad to N >= 4+nc to avoid layout-heuristic ambiguity
    pred = torch.cat([pred, torch.zeros(1, NC + 4, 6)], dim=2)

    dets = post_process(
        pred, orig_shapes=[(640, 640)],
        conf_thres=0.5, iou_thres=0.5, imgsz=640,
        ratios=[(1.0, 1.0)], pads=[(0, 0)],
    )
    assert len(dets) == 1
    assert len(dets[0]) == 2
    for d in dets[0]:
        assert d[4] >= 0.5
        assert d[5] == 0


def test_postprocess_ndarray_input_runs():
    boxes = [[0.5, 0.5, 0.4, 0.4]] + [[0.0, 0.0, 0.0, 0.0]] * 7
    cls = [[0.9, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 7
    pred = _make_pred(boxes, cls).numpy()
    dets = post_process(
        pred, orig_shapes=[(640, 640)],
        conf_thres=0.5, iou_thres=0.5, imgsz=640,
        ratios=[(1.0, 1.0)], pads=[(0, 0)],
    )
    assert len(dets) == 1
    assert len(dets[0]) == 1


def test_postprocess_list_input_runs():
    boxes = [[0.5, 0.5, 0.4, 0.4]] + [[0.0, 0.0, 0.0, 0.0]] * 7
    cls = [[0.9, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 7
    pred = _make_pred(boxes, cls)
    dets = post_process(
        [pred], orig_shapes=[(640, 640)],
        conf_thres=0.5, iou_thres=0.5, imgsz=640,
        ratios=[(1.0, 1.0)], pads=[(0, 0)],
    )
    assert len(dets) == 1
    assert len(dets[0]) == 1


def test_postprocess_clips_to_image_bounds():
    """Boxes far outside image bounds are clipped to [0, w-1]/[0, h-1]."""
    boxes = [[2.0, 2.0, 0.4, 0.4]] + [[0.0, 0.0, 0.0, 0.0]] * 7
    cls = [[0.95, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 7
    pred = _make_pred(boxes, cls)
    dets = post_process(
        pred, orig_shapes=[(100, 200)],
        conf_thres=0.3, iou_thres=0.5, imgsz=640,
        ratios=[(1.0, 1.0)], pads=[(0, 0)],
    )
    for x1, y1, x2, y2, conf, cls_id in dets[0]:
        assert 0 <= x1 < 200
        assert 0 <= y1 < 100
        assert 0 <= x2 < 200
        assert 0 <= y2 < 100


def test_postprocess_shape_mismatch_raises():
    pred = torch.zeros(1, 7, 8)
    with pytest.raises(ValueError):
        post_process(
            pred,
            orig_shapes=[(640, 640), (640, 640)],   # batch=1 but 2 orig shapes
            conf_thres=0.5, iou_thres=0.5, imgsz=640,
        )


def test_postprocess_low_conf_is_filtered():
    boxes = [[0.5, 0.5, 0.4, 0.4]] + [[0.0, 0.0, 0.0, 0.0]] * 7
    cls = [[0.1, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 7
    pred = _make_pred(boxes, cls)
    dets = post_process(
        pred, orig_shapes=[(640, 640)],
        conf_thres=0.5, iou_thres=0.5, imgsz=640,
        ratios=[(1.0, 1.0)], pads=[(0, 0)],
    )
    assert dets == [[]]


def test_postprocess_output_types():
    boxes = [[0.5, 0.5, 0.4, 0.4]] + [[0.0, 0.0, 0.0, 0.0]] * 7
    cls = [[0.9, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 7
    pred = _make_pred(boxes, cls)
    dets = post_process(
        pred, orig_shapes=[(640, 640)],
        conf_thres=0.5, iou_thres=0.5, imgsz=640,
        ratios=[(1.0, 1.0)], pads=[(0, 0)],
    )
    x1, y1, x2, y2, conf, cls_id = dets[0][0]
    assert isinstance(x1, int)
    assert isinstance(y1, int)
    assert isinstance(x2, int)
    assert isinstance(y2, int)
    assert isinstance(conf, float)
    assert isinstance(cls_id, int)
    assert round(conf, 4) == conf


# ---------------------------------------------------------------------------
# Regression tests for the _ensure_4nc_first layout heuristic.
# Pure torch — no model or GPU required.
# ---------------------------------------------------------------------------
def test_ensure_4nc_first_already_correct():
    """(bs, 4+nc, N) is returned untouched (contiguous copy at worst)."""
    from src.postprocess import _ensure_4nc_first

    pred = torch.zeros(1, 7, 100)  # nc=3 → 4+nc=7
    out = _ensure_4nc_first(pred, nc=3)
    assert out.shape == (1, 7, 100)


def test_ensure_4nc_first_permutes_when_flipped():
    """(bs, N, 4+nc) is permuted to (bs, 4+nc, N)."""
    from src.postprocess import _ensure_4nc_first

    pred = torch.zeros(1, 100, 7)
    out = _ensure_4nc_first(pred, nc=3)
    assert out.shape == (1, 7, 100)


def test_ensure_4nc_first_nc_resolves_4pnc_gt_N():
    """When num_boxes < 4+nc the magnitude heuristic would be wrong, but nc
    resolves it exactly. Layout (bs, N=5, 4+nc=90) → permute to (bs, 90, 5)
    rather than leave as-is."""
    from src.postprocess import _ensure_4nc_first

    pred = torch.zeros(1, 5, 90)  # nc=86 → 4+nc=90, "N"=5 (very small grid)
    out = _ensure_4nc_first(pred, nc=86)
    assert out.shape == (1, 90, 5)


def test_ensure_4nc_first_nc_raises_on_unrecognized_layout():
    """Neither dim matches 4+nc → explicit error (instead of silent ambiguity)."""
    from src.postprocess import _ensure_4nc_first

    pred = torch.zeros(1, 8, 9)  # neither 8 nor 9 equals 4+nc=15
    with pytest.raises(ValueError, match="Cannot resolve pred layout"):
        _ensure_4nc_first(pred, nc=11)


def test_ensure_4nc_first_heuristic_falls_back_when_nc_is_none():
    """``nc=None`` (default) exercises the magnitude-heuristic path."""
    from src.postprocess import _ensure_4nc_first

    # Standard YOLOv8s-ish: (bs, N=200, 4+nc=16). Magnitude > SMALL → permute.
    pred = torch.zeros(1, 200, 16)
    out = _ensure_4nc_first(pred, nc=None)
    assert out.shape == (1, 16, 200)
