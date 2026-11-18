"""Model-free tests for the QNN backend's optional-dependency surface.

The QNN backend executes a Qualcomm HTP context binary on a connected adb
device (see ``src/qnn_engine.py`` / docs/QNN.md). QAIRT + adb are *optional*
host dependencies — the base pin set does not include the SDK (a zip download,
not pip) and adb may be absent. These tests guard the contract the harnesses
rely on, without needing the SDK, a device, or any model artifact:

* the public names resolve through ``src``'s PEP-562 ``_LAZY`` map;
* the resolvers honor the ``QNN_SDK_ROOT`` / ``QNN_ADB`` env overrides and
  raise hint-carrying ``RuntimeError`` when the environment is incomplete;
* ``adb_device_serial`` honors ``QNN_DEVICE_SERIAL`` and reports "no device"
  when nothing is attached;
* the raw-layout helpers round-trip: NCHW→NHWC byte order (the graph-layout
  contract), and the split (box / cls) raw outputs re-assemble into the
  ``[bs, 4+nc, N]`` layout ``post_process`` / ``compare_tensors`` consume —
  including the NHWC→channel-first re-order the device output carries;
* ``cli.consistency._resolve_consistency_model`` passes a ``qnn:`` spec
  through verbatim;
* ``ModelWrapper`` detects the ``qnn:`` prefix lazily — no runner is built at
  construction; only an actual ``forward`` call resolves the host toolchain
  (and may raise the hint ``ImportError``);
* constructing ``QNNEngineRunner`` without the toolchain raises
  ``ImportError`` with the setup hint, never ``AttributeError`` (mirrors the
  openvino/tensorrt/trt_cpp optional-dep posture).

Heavy: imports ``src.consistency`` (-> ultralytics). No SDK / adb / device /
model required.
"""
import numpy as np
import pytest

import src
from src.consistency import ModelWrapper
from cli.consistency import _resolve_consistency_model
from src.qnn_engine import (
    adb_device_serial,
    load_split_raw_outputs,
    nchw_batch_to_nhwc_raws,
    resolve_adb,
    resolve_qnn_sdk_root as resolve_sdk_root_impl,
    split_outputs_to_4nc,
)


# ---------------------------------------------------------------------------
# _LAZY registration — the names must resolve through the package surface.
# ---------------------------------------------------------------------------
LAZY_NAMES = ["QNNEngineRunner", "qnn_available"]


def test_public_names_resolve_through_lazy():
    for name in LAZY_NAMES:
        assert hasattr(src, name), (
            f"{name!r} is not resolvable via src — missing from _LAZY?"
        )
        assert name in src._LAZY, f"{name!r} missing from src._LAZY map"


# ---------------------------------------------------------------------------
# Resolvers — env overrides + hint-carrying failures.
# ---------------------------------------------------------------------------
def test_resolve_sdk_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("QNN_SDK_ROOT", str(tmp_path))
    assert resolve_sdk_root_impl() == str(tmp_path)


def test_resolve_sdk_root_home_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    fake_home = tmp_path / "qairt" / "qairt"
    (fake_home / "2.40.1.251119").mkdir(parents=True)
    (fake_home / "2.42.0.251225").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    got = resolve_sdk_root_impl()
    # The latest version directory wins (sorted; a zip-layout install).
    assert got.endswith("2.42.0.251225")


def test_resolve_sdk_root_missing_raises_hint(monkeypatch):
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    monkeypatch.setattr(
        "pathlib.Path.home",
        lambda: __import__("pathlib").Path("/nonexistent-qnn-home"),
    )
    with pytest.raises(RuntimeError, match="QNN_SDK_ROOT"):
        resolve_sdk_root_impl()


def test_resolve_adb_env_override(tmp_path):
    fake = tmp_path / "adb.exe"
    fake.write_text("")
    from src.qnn_engine import resolve_adb as _r
    # Env override wins over PATH (which may or may not carry adb).
    import os
    old = os.environ.get("QNN_ADB")
    os.environ["QNN_ADB"] = str(fake)
    try:
        assert _r() == str(fake)
    finally:
        if old is None:
            os.environ.pop("QNN_ADB", None)
        else:
            os.environ["QNN_ADB"] = old


def test_resolve_adb_missing_raises_hint(monkeypatch):
    monkeypatch.delenv("QNN_ADB", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="adb not found"):
        resolve_adb()


def test_adb_device_serial_no_device_raises_hint(monkeypatch):
    monkeypatch.delenv("QNN_DEVICE_SERIAL", raising=False)
    monkeypatch.setattr(
        "src.qnn_engine.resolve_adb", lambda: "adb-stub",
    )

    class _Out:
        stdout = "List of devices attached\n"

    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: _Out(),
    )
    with pytest.raises(RuntimeError, match="no ADB device"):
        adb_device_serial()


def test_adb_device_serial_picks_device_state(monkeypatch):
    monkeypatch.delenv("QNN_DEVICE_SERIAL", raising=False)
    monkeypatch.setattr("src.qnn_engine.resolve_adb", lambda: "adb-stub")

    class _Out:
        stdout = (
            "List of devices attached\n"
            "abc123\tunauthorized\n"
            "d7aea176\tdevice\n"
        )

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Out())
    assert adb_device_serial() == "d7aea176"


def test_adb_device_serial_pin_filters(monkeypatch):
    monkeypatch.setenv("QNN_DEVICE_SERIAL", "abc123")
    monkeypatch.setattr("src.qnn_engine.resolve_adb", lambda: "adb-stub")

    class _Out:
        stdout = "List of devices attached\nd7aea176\tdevice\n"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Out())
    with pytest.raises(RuntimeError, match="abc123"):
        adb_device_serial()


# ---------------------------------------------------------------------------
# Raw layout helpers — the graph-layout contract, model-free.
# ---------------------------------------------------------------------------
def test_nchw_batch_to_nhwc_raws_byte_order(tmp_path):
    batch = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    paths = nchw_batch_to_nhwc_raws(batch, tmp_path)
    assert len(paths) == 2
    raw0 = np.fromfile(str(paths[0]), dtype=np.float32).reshape(4, 4, 3)
    # NHWC: value at (h, w, c) == batch[0, c, h, w]
    for h in range(4):
        for w in range(4):
            for c in range(3):
                assert raw0[h, w, c] == batch[0, c, h, w]


def test_split_outputs_reassemble_channel_first(tmp_path):
    nc, n = 3, 10
    box = np.random.rand(4, n).astype(np.float32)
    cls = np.random.rand(nc, n).astype(np.float32)
    d = tmp_path / "Result_0"
    d.mkdir()
    # Output raws come back channel-first (measured: NCHW planes, corr 0.9988
    # vs the NHWC-transpose reading's −0.111 against the ORT FP32 reference).
    box.ravel().tofile(str(d / "x_model_22_Mul_5_output_0.raw"))
    cls.ravel().tofile(str(d / "x_model_22_Sigmoid_output_0.raw"))
    got_box, got_cls = load_split_raw_outputs(d, nc, num_anchors=n)
    assert np.allclose(got_box, box)
    assert np.allclose(got_cls, cls)
    full = split_outputs_to_4nc([d], nc, num_anchors=n)
    assert full.shape == (1, 4 + nc, n)
    assert np.allclose(full[0, :4], box)
    assert np.allclose(full[0, 4:], cls)


def test_split_outputs_wrong_size_raises(tmp_path):
    nc, n = 3, 10
    d = tmp_path / "Result_0"
    d.mkdir()
    np.zeros(4 * n - 1, dtype=np.float32).tofile(str(d / "a_Mul_5_output_0.raw"))
    np.zeros(nc * n, dtype=np.float32).tofile(str(d / "a_Sigmoid_output_0.raw"))
    with pytest.raises(RuntimeError, match="box raw size"):
        load_split_raw_outputs(d, nc, num_anchors=n)


# ---------------------------------------------------------------------------
# Consistency prefix passthrough + lazy ModelWrapper detection.
# ---------------------------------------------------------------------------
def test_resolve_consistency_model_passthrough_qnn():
    spec = "qnn:models/yolov8s_w8a16.serialized.bin"
    assert _resolve_consistency_model(spec, "models/yolov8s.pt") == spec


def test_model_wrapper_qnn_prefix_is_lazy():
    w = ModelWrapper("qnn:models/yolov8s_w8a16.serialized.bin", "cpu")
    assert w.type == "qnn"
    assert w.model_path == "models/yolov8s_w8a16.serialized.bin"
    # No runner is built at construction (the lazy contract) — this works
    # without the SDK / adb / device present.
    assert w.model is None


def test_model_wrapper_qnn_forward_without_toolchain_raises_importerror(monkeypatch):
    w = ModelWrapper("qnn:models/yolov8s_w8a16.serialized.bin", "cpu")
    monkeypatch.setattr("src.qnn_engine.qnn_available", lambda: False)
    x = np.zeros((1, 3, 64, 64), dtype=np.float32)
    with pytest.raises(ImportError, match="QNN"):
        w.forward(__import__("torch").from_numpy(x))


# ---------------------------------------------------------------------------
# Engine construction guard.
# ---------------------------------------------------------------------------
def test_runner_raises_importerror_without_toolchain(monkeypatch):
    monkeypatch.setattr("src.qnn_engine.qnn_available", lambda: False)
    with pytest.raises(ImportError, match="QNN"):
        src.QNNEngineRunner(
            model_path="models/yolov8s_w8a16.serialized.bin",
            data_yaml="data/data.yaml",
        )
