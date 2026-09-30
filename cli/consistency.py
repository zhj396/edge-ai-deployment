from pathlib import Path

from utils import get_logger
from . import (
    ConsistencyConfig,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_FP32,
    DEFAULT_MODEL_PT,
    resolve_path_arg,
    run_timestamp,
)
from .consistency_tol import (
    needs_loose_pair,
    resolve_tolerances,
)
from src import validate_consistency

# Re-exported for back-compat / tests; the implementation lives in consistency_tol
# so it stays unit-testable without importing ``src`` (ultralytics).
_needs_loose_pair = needs_loose_pair
_resolve_tolerances = resolve_tolerances

logger = get_logger(__name__)


def _resolve_consistency_model(arg, default):
    """Resolve a --model1/--model2 arg, allowing a backend prefix.

    A plain path goes through ``resolve_path_arg`` (existing-validation). A
    prefixed spec (``ort_cpp:<onnx>`` / ``openvino(_int8):<xml>``) is a backend
    selector, not a path — pass it through verbatim; ``ModelWrapper`` parses
    the prefix and resolves the model lazily on the first forward.
    """
    if arg and str(arg).startswith(
        ("ort_cpp:", "tensorrt:", "trt_cpp:", "openvino:", "openvino_int8:")
    ):
        return arg
    return resolve_path_arg(arg, default)


# =========================================================
# Add CLI subparser
# =========================================================
def add_parser(subparsers):
    parser = subparsers.add_parser(
        "consistency",
        help="Validate consistency between two models",
        epilog=(
            "Example: python main.py consistency --model1 models/yolov8s.pt "
            "--model2 models/yolov8s_fp32.onnx --imgs-input data --mode tensor"
        ),
    )
    # ``default=None`` so argparse skips ``existing_validation`` on the documented defaults — see
    # cli/__init__.py for the rationale. The defaults are resolved inside ``run()``.

    parser.add_argument(
        "--model1", type=str, default=None,
        help=f"Reference model (default: {DEFAULT_MODEL_PT}). Prefix "
        f"'ort_cpp:' for the C++ ORT backend (requires the built ort_cpp exe; "
        f"see docs/ORT_CPP.md), 'tensorrt:' to compare a serialized .engine "
        f"via the TensorRT backend, e.g. "
        f"'tensorrt:models/yolov8s_fp16.engine'. Prefix 'trt_cpp:' to compare "
        f"the SAME .engine via the in-process C++ pybind11 backend (build it: "
        f"cmake --build cpp/build --target _trt_cpp). Prefix 'openvino:' or "
        f"'openvino_int8:' to compare an OpenVINO IR, e.g. "
        f"'openvino_int8:models/yolov8s_openvino_int8.xml' "
        f"(OPENVINO_DEVICE env var selects CPU/GPU/AUTO).",
    )
    parser.add_argument(
        "--model2", type=str, default=None,
        help=f"Comparison model (default: {DEFAULT_MODEL_FP32}). Accepts the "
        f"'ort_cpp:', 'tensorrt:', 'trt_cpp:', 'openvino:' and "
        f"'openvino_int8:' prefixes like --model1.",
    )
    parser.add_argument(
        "--imgs-input", type=Path, default=None,
        help=f"Image directory containing data.yaml (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument("--mode", type=str, choices=["tensor", "detection"], default="tensor")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--img-sizes", type=int, nargs="+", default=[640])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--resnet50", type=Path, default=None,
        help="Optional local ResNet50 checkpoint (default: torchvision auto-download)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Validation device (default: cpu)",
    )
    # Default 1e-3 / 1e-2 covers both PT↔ONNX-FP32 and FP32↔INT8 — INT8 routinely raises p99
    # above 0.05 via noise in low-confidence boxes (advisory tensor stats only), so the stricter
    # 1e-4 / 1e-3 default would false-fail INT8 runs. Tighten explicitly for PT-vs-FP32.
    parser.add_argument(
        # ``default=None`` so argparse never forces a value; ``run()`` auto-selects strict
        # (1e-4 / 1e-3) for PT↔FP32 and loose (1e-3 / 1e-2) when a reduced-precision side
        # (INT8 or FP16) is involved. Set both explicitly to override.
        "--atol", type=float, default=None,
        help="Absolute tolerance. Default: auto — 1e-4 (PT↔FP32) / 1e-3 (any INT8 or "
             "FP16 side).",
    )
    parser.add_argument(
        "--rtol", type=float, default=None,
        help="Relative tolerance. Default: auto — 1e-3 (PT↔FP32) / 1e-2 (any INT8 or "
             "FP16 side).",
    )
    parser.add_argument(
        "--report-path", type=Path, default="results/consistency_report.json",
        help="JSON report stem + directory (default: results/consistency_report.json). "
             "The CLI inserts a per-run timestamp between stem and suffix, so "
             "this path writes results/consistency_report_<TS>.json; distinct "
             "stems (e.g. results/consistency_tensor.json) keep tensor and "
             "detection runs side by side, and each run writes its own file. "
             "<TS> is fixed-width and sortable, so the newest report of a "
             "stem is selected by name sort (e.g. "
             "`ls results/consistency_tensor_*.json | sort | tail -1`).",
    )

    return parser


# =========================================================
# Run consistency validation command
# =========================================================
def run(args):
    model1 = _resolve_consistency_model(args.model1, DEFAULT_MODEL_PT)
    model2 = _resolve_consistency_model(args.model2, DEFAULT_MODEL_FP32)
    atol, rtol = resolve_tolerances(args.atol, args.rtol, model1, model2)
    logger.info(
        "Starting consistency check: %s vs %s | atol=%g rtol=%g%s",
        Path(str(model1)).name, Path(str(model2)).name, atol, rtol,
        " (auto: strict PT↔FP32)"
        if (args.atol is None and not needs_loose_pair(model1, model2))
        else (" (auto: loose, INT8/FP16 involved)" if args.atol is None else " (explicit)"),
    )
    # One timestamp per run (see cli.run_timestamp for the format rationale).
    # Insert <ts> between stem and suffix so the file is ``<stem>_<TS>.json``
    # — the documented format and the fail_dir name-replace derivation in
    # src/consistency.py both anchor on the suffixed name.
    ts = run_timestamp()
    report_path = args.report_path.with_name(
        f"{args.report_path.stem}{ts}{args.report_path.suffix}"
    )
    cfg = ConsistencyConfig(
        model1=model1,
        model2=model2,
        imgs_input=resolve_path_arg(args.imgs_input, DEFAULT_DATA_DIR),
        mode=args.mode,
        max_images=args.max_images,
        img_sizes=args.img_sizes,
        batch_sizes=args.batch_sizes,
        resnet50=args.resnet50,
        device=args.device,
        atol=atol,
        rtol=rtol,
        report_path=report_path,
    )
    results = validate_consistency(
        model1=cfg.model1,
        model2=cfg.model2,
        imgs_input=cfg.imgs_input,
        mode=cfg.mode,
        max_images=cfg.max_images,
        img_sizes=cfg.img_sizes,
        batch_sizes=cfg.batch_sizes,
        resnet50=cfg.resnet50,
        device=cfg.device,
        atol=cfg.atol,
        rtol=cfg.rtol,
        report_path=cfg.report_path,
    )
    logger.info("Consistency report: %s", report_path)
    logger.info("Consistency validation complete")
    # Propagate pass/fail to the process exit code so CI (or any `set -e`
    # driver script) can act on a failed consistency check.
    return 0 if results.get("overall_pass") else 1
