"""Unit tests for src/quantize.py calibration-reader helpers.

Pure-logic coverage of the calibration data path: image readability filtering
and reader construction — no ONNX model or quantization run required. ``src.quantize``
is imported inside each test so module collection stays lightweight. The
hash-seed determinism test at the bottom is the one exception: it needs the
real ``data/`` split and skips (via the ``data_dir`` fixture) without it.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _write_minimal_jpeg(path):
    """Tiny valid 1×1 JPEG via Pillow — a real encode is required because
    PIL's ``verify()`` rejects bare SOI/EOI marker pairs.
    """
    from PIL import Image

    Image.new("RGB", (1, 1), (0, 0, 0)).save(str(path), "JPEG")


def test_filter_readable_keeps_valid_drops_garbage(tmp_path):
    from src.quantize import _filter_readable

    good = tmp_path / "ok.jpg"
    _write_minimal_jpeg(good)

    garbage = tmp_path / "broken.jpg"
    garbage.write_bytes(b"not a real jpeg")

    missing = tmp_path / "no_such.jpg"
    # missing intentionally not created

    dirs = tmp_path / "a_dir.jpg"
    dirs.mkdir()  # directory with .jpg suffix — should be rejected

    kept, dropped = _filter_readable([good, garbage, missing, dirs])
    assert kept == [good]
    assert len(dropped) == 3
    dropped_paths = [p for p, _ in dropped]
    assert garbage in dropped_paths
    assert missing in dropped_paths
    assert dirs in dropped_paths


def test_filter_readable_empty_input():
    from src.quantize import _filter_readable

    kept, dropped = _filter_readable([])
    assert kept == []
    assert dropped == []


def test_yolov8_calibration_reader_raises_on_all_unreadable(tmp_path):
    """When every supplied image is unreadable, the constructor raises a
    clear error."""
    from src.quantize import YOLOv8CalibrationDataReader

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"garbage")

    with pytest.raises(RuntimeError, match="No readable calibration images"):
        YOLOv8CalibrationDataReader(
            calibration_imgs=[bad], imgsz=640, batch_size=1,
        )


def test_yolov8_calibration_reader_raises_when_every_batch_fails(tmp_path, monkeypatch):
    """Header-readable images whose pixel decode fails at preprocessing are
    skipped per batch; a run where no batch succeeds raises."""
    import src.quantize as quantize_module
    from src.quantize import YOLOv8CalibrationDataReader

    imgs = []
    for i in range(3):
        p = tmp_path / f"ok_{i}.jpg"
        _write_minimal_jpeg(p)
        imgs.append(p)

    def _fail(**kwargs):
        raise RuntimeError("All images failed preprocessing")

    monkeypatch.setattr(quantize_module, "preprocess_imgs", _fail)

    reader = YOLOv8CalibrationDataReader(
        calibration_imgs=imgs, imgsz=640, batch_size=2,
    )
    assert len(reader) == 2  # ceil(3/2) batches planned

    with pytest.raises(RuntimeError, match="nothing to calibrate on"):
        reader.get_next()


def test_calibration_sampler_rejects_nonpositive_size():
    """calibration_size <= 0 is a construction-time error, checked before any
    dataset access."""
    from src.sampler import CalibrationSampler

    with pytest.raises(ValueError, match="calibration_size must be positive"):
        CalibrationSampler(data_yaml="unused.yaml", calibration_size=0, device="cpu")


# Driver for the cross-process determinism test below. Bypasses __init__
# (which loads ResNet-50) and runs only the dataset load + stratified stage —
# the stage whose input order depends on PYTHONHASHSEED.
_STRAT_DRIVER = """
import json, random
from pathlib import Path
import numpy as np
from src.sampler import CalibrationSampler

random.seed(42)
np.random.seed(42)
s = CalibrationSampler.__new__(CalibrationSampler)
s.data_yaml = Path({yaml!r})
s.calibration_size = 300
s._load_dataset()
print(json.dumps([str(p) for p in s._stratified_sampling()]))
"""


def test_stratified_sampling_is_hash_seed_deterministic(data_dir):
    """Verify stratified sampling returns identical selections under
    different PYTHONHASHSEED values (data-gated: skips without the val
    split)."""
    root = Path(__file__).resolve().parent.parent
    driver = _STRAT_DRIVER.format(yaml=str(data_dir / "data.yaml"))

    outputs = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-c", driver],
            capture_output=True, text=True, cwd=str(root), env=env,
        )
        assert proc.returncode == 0, proc.stderr
        outputs.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    assert outputs[0], "stratified sampling selected nothing"
    assert outputs[0] == outputs[1], (
        "selection differs across PYTHONHASHSEED — sampler is not reproducible"
    )
