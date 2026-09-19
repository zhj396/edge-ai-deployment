"""Tests for the array-backed preprocessing path (``preprocess_frames``).

The array-backed path exists so the FastAPI server can feed already-decoded
frames straight into the engine. Its defining property: it must produce the
exact same preprocessed batch as the file-backed path — the two share
``_preprocess_bgr`` + ``_assemble_batch``, so these tests pin that parity
(plus the input-normalization and batch-contract details).
"""
import os

import cv2
import numpy as np
import pytest

from src.preprocess import preprocess_frames, preprocess_single


def _synthetic_bgr(h: int = 480, w: int = 640) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_preprocess_frames_batch_contract():
    frame = _synthetic_bgr()
    data = preprocess_frames([frame, frame], imgsz=640)

    # Same batch layout as preprocess_imgs: (B, 3, H, W) float32 normalized to [0, 1]
    assert tuple(data["images"].shape) == (2, 3, 640, 640)
    assert str(data["images"].dtype) == "torch.float32"
    assert float(data["images"].max()) <= 1.0

    # orig_imgs only populated when original=True (same contract as preprocess_imgs)
    assert data["orig_imgs"] == []
    assert len(data["orig_shapes"]) == 2
    assert len(data["ratios"]) == 2
    assert len(data["pads"]) == 2
    assert data["paths"] == ["<frame:0>", "<frame:1>"]


def test_preprocess_frames_matches_file_backed_path():
    """Array path must be pixel-identical to the file path for the same image."""
    frame = _synthetic_bgr()
    tmpdir = os.path.join(os.path.dirname(__file__), "_tmp_parity")
    os.makedirs(tmpdir, exist_ok=True)
    path = os.path.join(tmpdir, "parity.png")
    try:
        cv2.imwrite(path, frame)  # PNG is lossless -> file decodes back to `frame`
        decoded = cv2.imread(path)

        from_file = preprocess_single(path, imgsz=640)
        from_arr = preprocess_frames([decoded], imgsz=640)

        expected = np.round(
            from_arr["images"][0].numpy() * 255
        ).astype(np.uint8)
        assert np.array_equal(from_file["img"], expected)
        assert from_file["ratio"] == from_arr["ratios"][0]
        assert from_file["pad"] == from_arr["pads"][0]
    finally:
        if os.path.exists(path):
            os.remove(path)
        os.rmdir(tmpdir)


def test_preprocess_frames_normalizes_gray_and_bgra():
    """Grayscale and BGRA frames are converted to BGR by the shared core."""
    gray = _synthetic_bgr()[:, :, 0]
    bgra = np.dstack([
        _synthetic_bgr(),
        np.full((480, 640), 255, dtype=np.uint8),
    ])
    for frames in ([gray], [bgra]):
        data = preprocess_frames(frames, imgsz=640)
        assert tuple(data["images"].shape) == (1, 3, 640, 640)


def test_preprocess_frames_empty_list_raises():
    with pytest.raises(ValueError):
        preprocess_frames([], imgsz=640)


def test_preprocess_frames_original_keeps_unletterboxed_copy():
    frame = _synthetic_bgr()
    data = preprocess_frames([frame], imgsz=640, original=True)
    assert len(data["orig_imgs"]) == 1
    assert data["orig_imgs"][0].shape == frame.shape
    # The kept original is the un-letterboxed input, not the letterboxed one
    assert np.array_equal(data["orig_imgs"][0], frame)
