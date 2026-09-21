"""Model-free tests for the OpenVINO backend's optional-dependency surface.

OpenVINO + NNCF are *optional* — the base pin set (requirements-cpu.txt)
does not include them. These tests guard the contract that the rest of the
toolchain relies on without needing openvino installed:

* the new public names resolve through ``src``'s PEP-562 ``_LAZY`` map (an
  unregistered name would raise ``AttributeError`` at ``from src import
  OpenVINOEngine``);
* the ``cli.openvino`` subcommand module imports cleanly and exposes
  ``add_parser`` / ``run`` (it does ``from src import ...`` at module top,
  so a broken ``_LAZY`` entry would surface here);
* the ``OpenVINO*Config`` dataclasses construct;
* when openvino/nncf are absent, constructing ``OpenVINOEngine`` and calling
  the convert/quantize entry points raise ``ImportError`` with a helpful
  message rather than ``AttributeError`` / ``NameError``;
* the static-batch padding contract (``_pad_to_static_batch``, shared by the
  sync and async paths) via a stub instance — model-free;
* the device preflight (``validate_device_request``): an unservable --device
  request resolves to an explicit, warned CPU fallback (or fails at engine
  init with an actionable message when no fallback is possible) — pure
  function, model-free;
* (opt-in, needs the pinned nncf) the NNCF SmoothQuant model_type gate that
  ``nncf_quantize_openvino``'s ``--smooth-quant`` flag depends on.

No ``.pt`` / ``.onnx`` / GPU required — runs on any laptop in <1 s.
"""
import numpy as np
import pytest

import src
from cli import openvino as cli_openvino
from cli import (
    OpenVINOConvertConfig,
    OpenVINOQuantizeConfig,
    OpenVINORunConfig,
)

# Installed-state flags used by the skipif guards below. Resolved once at
# import time through src's PEP-562 lazy map (which itself is part of what
# this module guards).
OPENVINO_INSTALLED = src.openvino_available()
NNCF_INSTALLED = src.nncf_available()


# ---------------------------------------------------------------------------
# _LAZY registration — the names must resolve through the package surface.
# ---------------------------------------------------------------------------
LAZY_NAMES = [
    "convert_onnx_to_openvino_ir",
    "nncf_quantize_openvino",
    "openvino_conversion_available",
    "nncf_available",
    "OpenVINOEngine",
    "OpenVINOAsyncEngine",
    "openvino_available",
]


def test_public_names_resolve_through_lazy():
    """Each name is reachable via ``getattr(src, name)`` (PEP 562 ``_LAZY``).

    An unregistered name would surface as ``AttributeError`` at import time
    and break ``cli/openvino.py``'s top-level ``from src import ...``.
    """
    for name in LAZY_NAMES:
        assert hasattr(src, name), (
            f"{name!r} is not resolvable via src — missing from _LAZY?"
        )
        assert name in src._LAZY, f"{name!r} missing from src._LAZY map"


def test_cli_openvino_exposes_parser_and_run():
    """The subcommand module imports without error and is CLI-shaped."""
    assert callable(getattr(cli_openvino, "add_parser", None))
    assert callable(getattr(cli_openvino, "run", None))


# ---------------------------------------------------------------------------
# Dataclasses construct with the fields the CLI populates.
# ---------------------------------------------------------------------------
def test_openvino_convert_config_constructs():
    cfg = OpenVINOConvertConfig(
        model="a.onnx", output="a.xml", imgsz=640, fp16=True,
    )
    assert cfg.fp16 is True


def test_class_names_from_data_yaml(tmp_path):
    """_class_names_from_data_yaml parses the YOLO names field (list or dict).

    Guards the IR class-name path: an OpenVINO IR loses the ultralytics
    'names' metadata, so the engine reads names from data.yaml instead.
    Model-free (the helper is pure; importing src.openvino_engine without
    openvino installed is guarded at module top).
    """
    from src.openvino_engine import _class_names_from_data_yaml
    # list form (the project's data.yaml uses a list)
    y = tmp_path / "data.yaml"
    y.write_text("names:\n- person\n- bicycle\n- car\nnc: 3\n")
    assert _class_names_from_data_yaml(y) == {0: "person", 1: "bicycle", 2: "car"}
    # dict form
    y2 = tmp_path / "d2.yaml"
    y2.write_text("names:\n  0: person\n  1: car\n")
    assert _class_names_from_data_yaml(y2) == {0: "person", 1: "car"}
    # missing names field -> None (engine falls back to class_N)
    y3 = tmp_path / "d3.yaml"
    y3.write_text("nc: 3\n")
    assert _class_names_from_data_yaml(y3) is None


def test_openvino_quantize_config_constructs():
    cfg = OpenVINOQuantizeConfig(
        model="a.onnx", output="a.xml", imgs_input="data",
        imgsz=640, max_cal_samples=500, subset_size=64, resnet50=None,
        smooth_quant=False,
    )
    assert cfg.resnet50 is None
    assert cfg.subset_size == 64


def test_openvino_run_config_constructs():
    cfg = OpenVINORunConfig(
        model="a.xml", imgs_input="data", imgsz=640, device="CPU",
        num_streams="AUTO", max_imgs=32, batch_size=8,
        conf=0.25, iou=0.45, output_dir="results/predictions",
    )
    assert cfg.device == "CPU"


# ---------------------------------------------------------------------------
# Batch-stepping contract (model-free: only _model_batch + logger are touched,
# so a stub instance via __new__ exercises it without openvino installed).
# ---------------------------------------------------------------------------
def _stub_engine(model_batch):
    from src.openvino_engine import OpenVINOEngine
    eng = object.__new__(OpenVINOEngine)
    eng._model_batch = model_batch
    return eng


def test_effective_batch_dynamic_ir_honors_request():
    eng = _stub_engine(None)
    assert eng._effective_batch(8) == 8
    assert eng._effective_batch(0) == 1  # clamped, never a zero-step loop


def test_effective_batch_static_ir_always_steps_at_model_batch():
    """A static-batch IR's DFL reshape only accepts its baked count, so the
    step must equal _model_batch in BOTH directions (over- and under-request);
    the trailing partial batch is zero-padded inside _forward."""
    eng = _stub_engine(1)
    assert eng._effective_batch(8) == 1
    eng = _stub_engine(4)
    assert eng._effective_batch(2) == 4  # under-request must not under-feed
    assert eng._effective_batch(8) == 4  # over-request must not over-feed


def test_pad_to_static_batch_pads_tail_and_reports_real_n():
    """Static IR + partial tail batch -> zero-padded to _model_batch, and
    real_n lets callers slice outputs back (no image dropped, no partial
    batch reaches the graph). Shared by sync _forward and async start_batch,
    so this stub covers both paths' padding contract."""
    eng = _stub_engine(4)
    batch = np.ones((2, 3, 8, 8), dtype=np.float32)
    padded, real_n = eng._pad_to_static_batch(batch)
    assert padded.shape[0] == 4
    assert real_n == 2
    assert np.all(padded[:2] == 1.0)  # real images untouched
    assert np.all(padded[2:] == 0.0)  # tail zero-padded


def test_pad_to_static_batch_passthrough_for_full_and_dynamic():
    """Full batches and dynamic IRs take the pass-through path (no copy)."""
    eng = _stub_engine(4)
    full = np.ones((4, 3, 8, 8), dtype=np.float32)
    padded, real_n = eng._pad_to_static_batch(full)
    assert padded is full and real_n == 4
    eng = _stub_engine(None)
    padded, real_n = eng._pad_to_static_batch(full[:2])
    assert padded.shape[0] == 2 and real_n == 2  # dynamic IR: any batch ok


# ---------------------------------------------------------------------------
# Device preflight (model-free pure function): a request OpenVINO cannot
# serve on this host resolves to an explicit CPU fallback (warned, never
# silent), or fails at engine init with an actionable message when the
# fallback is impossible / disabled — not as an opaque C++ compile_model
# exception later.
# ---------------------------------------------------------------------------
def test_validate_device_request_accepts_servable_requests():
    from src.openvino_engine import validate_device_request

    available = ["CPU", "GPU"]
    # All of these must resolve to the requested device unchanged.
    assert validate_device_request("CPU", available) == "CPU"
    assert validate_device_request("GPU", available) == "GPU"
    # AUTO dispatches over enumerated devices ... even with nothing listed.
    assert validate_device_request("AUTO", ["CPU"]) == "AUTO"
    assert validate_device_request("AUTO", []) == "AUTO"
    # A compound form is only servable when EVERY named component is
    # enumerated — a partially-available compound (e.g. MULTI:CPU,GPU with
    # no GPU) must resolve here, not die at compile_model with OpenVINO's
    # opaque C++ error.
    assert validate_device_request("MULTI:CPU,GPU", available) == "MULTI:CPU,GPU"
    assert validate_device_request("HETERO:GPU,CPU", available) == "HETERO:GPU,CPU"


def test_validate_device_request_falls_back_to_cpu_explicitly():
    """Unservable request + CPU enumerated -> resolved device is CPU.

    This is the exact WSL2-no-GPU-driver case: --device GPU on a host where
    OpenVINO only sees ['CPU'] must still run (on CPU), loudly — the warning
    carries the device list and the driver hint so a benchmark run can never
    silently claim GPU numbers that were measured on CPU.
    """
    from src.openvino_engine import validate_device_request

    available = ["CPU"]
    assert validate_device_request("GPU", available) == "CPU"
    # Compounds also fall back — both with zero present components and with
    # only some (every named component must be enumerated to keep the form).
    assert validate_device_request("MULTI:GPU,NPU", available) == "CPU"
    assert validate_device_request("HETERO:GPU,NPU", available) == "CPU"
    assert validate_device_request("MULTI:CPU,GPU", available) == "CPU"
    assert validate_device_request("HETERO:GPU,CPU", available) == "CPU"
    assert validate_device_request("MYRIAD", available) == "CPU"
    # A custom fallback is honored when enumerated.
    assert validate_device_request("NPU", ["CPU", "GPU"], fallback="GPU") == "GPU"


def test_validate_device_request_fallback_is_warned_with_hint(caplog):
    """The fallback must be explicit: a warning naming request, devices,
    and the GPU driver hint (/dev/dxg) — never a silent downgrade."""
    import logging
    from src.openvino_engine import validate_device_request

    with caplog.at_level(logging.WARNING, logger="src.openvino_engine"):
        resolved = validate_device_request("GPU", ["CPU"])
    assert resolved == "CPU"
    assert "Falling back" in caplog.text or "falling back" in caplog.text
    assert "dxg" in caplog.text


def test_validate_device_request_rejects_when_no_fallback_possible():
    from src.openvino_engine import validate_device_request

    # Strict mode (fallback=None) preserves reject-on-absent-device.
    available = ["CPU"]
    with pytest.raises(RuntimeError, match="not available"):
        validate_device_request("GPU", available, fallback=None)
    # The GPU branch must carry the driver hint (WSL2 /dev/dxg).
    with pytest.raises(RuntimeError, match="dxg"):
        validate_device_request("GPU", available, fallback=None)
    with pytest.raises(RuntimeError, match="unavailable"):
        validate_device_request("MULTI:GPU,NPU", available, fallback=None)
    with pytest.raises(RuntimeError, match="not available"):
        validate_device_request("MYRIAD", available, fallback=None)
    with pytest.raises(RuntimeError, match="not available"):
        validate_device_request("FOO:CPU", available, fallback=None)
    # Default fallback mode still raises when CPU itself is absent or is
    # the (unservable) request — nothing to fall back to.
    with pytest.raises(RuntimeError, match="not available"):
        validate_device_request("GPU", [])
    with pytest.raises(RuntimeError, match="not available"):
        validate_device_request("CPU", ["GPU"])
    # A fallback that isn't enumerated can't rescue the request either.
    with pytest.raises(RuntimeError, match="not available"):
        validate_device_request("GPU", ["CPU"], fallback="NPU")


# ---------------------------------------------------------------------------
# NNCF SmoothQuant gate (opt-in — requires the pinned nncf; skipped otherwise).
# Root-cause guard for the --smooth-quant no-op bug: on nncf 3.3.0 the PTQ
# pipeline adds SmoothQuant ONLY for model_type=TRANSFORMER, so
# nncf_quantize_openvino must pass model_type (+ a pinned preset) when the
# flag is on. If a future NNCF moves the gate, this test fails and the
# sq_kwargs block in src/openvino_convert.py should be revisited.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not NNCF_INSTALLED, reason="nncf not installed")
def test_smooth_quant_is_gated_on_transformer_model_type():
    import inspect

    try:
        from nncf.quantization.algorithms.post_training.pipeline import (
            create_ptq_pipeline,
        )
    except ImportError:
        pytest.skip("nncf internals moved; re-anchor the SmoothQuant guard")
    import nncf

    def pipeline_algos(model_type):
        p = create_ptq_pipeline(
            model_type=model_type,
            subset_size=8,
            advanced_parameters=nncf.AdvancedQuantizationParameters(
                smooth_quant_alphas=nncf.AdvancedSmoothQuantParameters(
                    matmul=0.95, convolution=0.95,
                ),
            ),
        )
        return {
            type(a).__name__ for step in p.pipeline_steps for a in step
        }

    # The bug this guards: alphas alone (default model_type=None) -> no SQ step.
    assert "SmoothQuant" not in pipeline_algos(None)
    # The fix: model_type=TRANSFORMER activates it.
    assert "SmoothQuant" in pipeline_algos(nncf.ModelType.TRANSFORMER)

    # The top-level kwargs nncf_quantize_openvino relies on must still exist.
    params = inspect.signature(nncf.quantize).parameters
    for kwarg in ("model_type", "preset", "fast_bias_correction"):
        assert kwarg in params, f"nncf.quantize lost its {kwarg!r} kwarg"


# ---------------------------------------------------------------------------
# Optional-dependency contract: absent openvino/nncf -> ImportError, not crash.
# These are skipped (not failed) when the packages ARE installed, so the suite
# stays green on a box that happens to have openvino.
# (OPENVINO_INSTALLED / NNCF_INSTALLED are defined at the top of the module.)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(OPENVINO_INSTALLED, reason="openvino is installed; "
                    "ImportError path cannot be exercised")
def test_engine_raises_importerror_without_openvino():
    OpenVINOEngine = src.OpenVINOEngine
    with pytest.raises(ImportError, match="openvino"):
        OpenVINOEngine(model_path="missing.xml", device="CPU")


@pytest.mark.skipif(OPENVINO_INSTALLED, reason="openvino is installed")
def test_convert_raises_importerror_without_openvino():
    with pytest.raises(ImportError, match="OpenVINO"):
        src.convert_onnx_to_openvino_ir(
            onnx_path="missing.onnx", output_xml="out.xml",
        )


@pytest.mark.skipif(NNCF_INSTALLED, reason="nncf is installed")
def test_quantize_raises_importerror_without_nncf():
    with pytest.raises(ImportError, match="NNCF"):
        src.nncf_quantize_openvino(
            onnx_path="missing.onnx", output_xml="out.xml",
            data_yaml="data/data.yaml",
        )
