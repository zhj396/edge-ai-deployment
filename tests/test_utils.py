"""Unit tests for utils helpers — no model or GPU required."""
import numpy as np
import pytest

from utils.comparison import cosine_similarity
from utils import (
    compare_models,
    existing_validation,
    file_size_mb,
    percentiles,
    sha256_of_file,
    summarize_runs,
)
from utils.visualization import color_for_class, draw_detections


# ---------------------------------------------------------------------------
def test_cosine_similarity_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-9)


def test_cosine_similarity_zero_vector():
    a = np.zeros(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_negative():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-9)


# ---------------------------------------------------------------------------
def test_percentiles_empty():
    out = percentiles([], qs=(50, 95, 99))
    assert all(v == 0.0 for v in out.values())


def test_percentiles_basic():
    out = percentiles(list(range(1, 101)), qs=(50, 95, 99))
    assert out["p50"] == pytest.approx(50.5, abs=1.0)
    assert out["p99"] == pytest.approx(99.01, abs=1.0)


def test_percentiles_accepts_generator():
    """Generator input (no ``len()``) is consumed directly via np.fromiter
    rather than falling into the n=0 short-circuit that returns all zeros."""
    gen = (x for x in [1.0, 2.0, 3.0, 4.0, 5.0])
    out = percentiles(gen)
    assert out["p50"] == pytest.approx(3.0)
    assert out["p99"] == pytest.approx(4.96, abs=1e-6)


def test_summarize_runs():
    out = summarize_runs([10.0, 20.0, 30.0, 40.0, 50.0])
    assert out["min"] == 10.0
    assert out["max"] == 50.0
    assert out["mean"] == 30.0
    assert "p99" in out


# ---------------------------------------------------------------------------
def test_color_for_class_is_stable():
    c1 = color_for_class(7)
    c2 = color_for_class(7)
    assert c1 == c2
    assert len(c1) == 3
    assert all(0 <= v <= 255 for v in c1)


def test_draw_detections_no_detections_returns_input():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(img, [])
    assert np.array_equal(out, img)


def test_draw_detections_draws_box():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(
        img, [(10, 10, 50, 50, 0.9, 0)],
        class_names={0: "person"},
    )
    # Modified image (some pixels drawn)
    assert out is not None
    assert out.shape == img.shape
    # At least one non-zero pixel was painted
    assert (out != 0).any()


# ---------------------------------------------------------------------------
def test_existing_validation_rejects_missing(tmp_path):
    with pytest.raises(ValueError):
        existing_validation(tmp_path / "nope.jpg")


def test_existing_validation_accepts_image(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")  # tiny valid JPEG
    assert existing_validation(img) == img


def test_existing_validation_accepts_dir(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    assert existing_validation(d) == d


def test_existing_validation_rejects_unknown_suffix(tmp_path):
    f = tmp_path / "x.xyz"
    f.write_text("hello")
    with pytest.raises(ValueError):
        existing_validation(f)


def test_existing_validation_missing_artifact_path_hints_artifacts_md(tmp_path):
    """A missing path under models/ or data/ points at ARTIFACTS.md — those
    directories are git-ignored, so the usual cause is a skipped download."""
    missing = tmp_path / "models" / "yolov8s.pt"
    with pytest.raises(ValueError, match="ARTIFACTS.md"):
        existing_validation(missing)


def test_existing_validation_missing_plain_path_has_no_artifact_hint(tmp_path):
    """Paths outside models//data/ keep the plain 'does not exist' message."""
    with pytest.raises(ValueError) as excinfo:
        existing_validation(tmp_path / "nope.jpg")
    assert "ARTIFACTS.md" not in str(excinfo.value)


def test_path_validation_error_is_visible_to_argparse():
    """argparse prints our message verbatim only for ArgumentTypeError;
    ValueError alone collapses into 'invalid existing_validation value'.
    The dual base keeps both the library contract and the CLI message."""
    import argparse

    from utils import PathValidationError

    assert issubclass(PathValidationError, argparse.ArgumentTypeError)
    assert issubclass(PathValidationError, ValueError)


def test_existing_validation_accepts_openvino_ir(tmp_path):
    # OpenVINO IR comes as .xml + sibling .bin; both must validate so the
    # `openvino run --model` path isn't rejected as an "unsupported format".
    xml = tmp_path / "y.xml"
    binf = tmp_path / "y.bin"
    xml.write_text("<net/>")
    binf.write_bytes(b"\0" * 4)
    assert existing_validation(xml) == xml
    assert existing_validation(binf) == binf


# ---------------------------------------------------------------------------
def test_sha256_of_file(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello")
    h = sha256_of_file(f)
    # sha256 of "hello"
    assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_file_size_mb(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"\x00" * (1024 * 1024))  # 1 MiB
    assert file_size_mb(f) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
def test_compare_models_identical(tmp_path):
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"same")
    f2.write_bytes(b"same")
    comp = compare_models(f1, f2)
    assert comp["identical"] is True
    assert comp["size_ratio"] == pytest.approx(1.0)


def test_compare_models_different(tmp_path):
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"abc")
    f2.write_bytes(b"defg")
    comp = compare_models(f1, f2)
    assert comp["identical"] is False
    assert comp["size_ratio"] != 1.0


# ---------------------------------------------------------------------------
def test_version_consistency_across_packages():
    """src / cli / utils all re-export the same ``__version__`` from src._version.

    Catches drift if someone hard-codes ``__version__ = "1.0.0"`` back into any
    of the three ``__init__.py`` files. The test fails the build before the
    version banner in ``main.py`` and the package ``__version__`` attributes
    can disagree in production.
    """
    import src  # noqa: F401  (import-side-effect: src.__version__ set)
    import cli  # noqa: F401  (cli re-exports from src)
    import utils  # noqa: F401  (utils re-exports from src)
    assert src.__version__ == cli.__version__ == utils.__version__
    assert src.__version__ == "1.0.0"
    # And the single source of truth should also agree.
    from src._version import __version__ as canonical
    assert canonical == src.__version__


# ---------------------------------------------------------------------------
# Regression tests: CLI path resolution and validation helpers —
# resolve_model_arg fallback semantics and the onnx_only suffix guard.
# ---------------------------------------------------------------------------
def test_resolve_model_arg_returns_value_when_supplied(tmp_path):
    """User-supplied Path is returned untouched (still validated upstream)."""
    from cli import resolve_model_arg

    p = tmp_path / "m.onnx"
    p.write_bytes(b"x")
    assert resolve_model_arg(p, default="/never/used/here") == p


def test_resolve_model_arg_falls_back_to_default(tmp_path):
    """When ``value is None`` and the default points at a real file, validate it."""
    from cli import resolve_model_arg

    p = tmp_path / "fp32.onnx"
    p.write_bytes(b"x")
    out = resolve_model_arg(None, default=str(p))
    assert out == p


def test_resolve_model_arg_default_missing_raises(tmp_path):
    """A missing default path surfaces from the workflow code
    (existing_validation) with a clean ValueError — never as a cryptic
    argparse parse-time error."""
    from cli import resolve_model_arg

    missing = tmp_path / "no_such_model.onnx"
    with pytest.raises(ValueError, match="does not exist"):
        resolve_model_arg(None, default=str(missing))


def test_onnx_only_accepts_onnx(tmp_path):
    """``inspect``'s --model/--compare reject non-ONNX suffixes (.pt/.pth and
    anything else) before they reach ``inspect_onnx``, which expects ONNX
    payloads."""
    from cli.inspect import onnx_only

    good = tmp_path / "x.onnx"
    good.write_bytes(b"\x08\x07")  # any non-empty bytes; suffix is what we check
    assert onnx_only(good) == good


def test_onnx_only_rejects_pt_file(tmp_path):
    from cli.inspect import onnx_only

    bad = tmp_path / "y.pt"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match=r"\.onnx"):
        onnx_only(bad)


def test_onnx_only_rejects_pt_with_unknown_suffix(tmp_path):
    from cli.inspect import onnx_only

    bad = tmp_path / "y.weird"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match=r"\.onnx"):
        onnx_only(bad)


def test_onnx_only_error_survives_argparse(tmp_path):
    """onnx_only raises PathValidationError (not plain ValueError) so
    `inspect --model x.pt` shows the '.onnx expected' reason instead of
    argparse's generic 'invalid onnx_only value' wrapper."""
    from cli.inspect import onnx_only
    from utils import PathValidationError

    bad = tmp_path / "y.pt"
    bad.write_bytes(b"x")
    with pytest.raises(PathValidationError, match=r"\.onnx"):
        onnx_only(bad)


def test_clamp_workers_passes_through_when_safe():
    """A reasonable request is honored."""
    from utils.threading import clamp_workers

    assert clamp_workers(2, ncpu=8) == 2
    assert clamp_workers(16, ncpu=16) == 16  # at the cap


def test_clamp_workers_caps_at_two_cpu():
    """Anything beyond 2×CPU is clamped (and logs a warning, which pytest
    captures silently)."""
    from utils.threading import clamp_workers

    # 64 workers on a 4-CPU box → clamped to 8.
    assert clamp_workers(64, ncpu=4) == 8
    # Very small ncpu → still ≥ 1.
    assert clamp_workers(1, ncpu=1) == 1


def test_clamp_workers_uses_real_ncpu_when_unspecified():
    """``ncpu=None`` falls back to ``os.cpu_count()`` (no cap test against a
    specific number — just that it returns a sane value)."""
    from utils.threading import clamp_workers

    out = clamp_workers(2)  # ncpu=None
    assert out >= 1
