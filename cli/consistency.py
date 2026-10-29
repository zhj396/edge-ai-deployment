from pathlib import Path

from utils import existing_validation, get_logger
from . import (
    ConsistencyConfig,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_FP32,
    DEFAULT_MODEL_PT,
    run_timestamp,
)
from src import validate_consistency

logger = get_logger(__name__)


def _resolve_consistency_model(arg, default):
    """Resolve a --model1/--model2 arg, allowing a backend prefix.

    A prefixed spec (``ort_cpp:<onnx>`` / ``tensorrt:<engine>`` /
    ``openvino(_int8):<xml>``) is a backend selector, not a path — pass it
    through verbatim; ``ModelWrapper`` parses the prefix. ort_cpp/tensorrt
    resolve their engine lazily on the first forward; openvino(_int8)
    path-checks the .xml at construction. A plain value is passed through
    unchanged; the existing-validated default applies only when the flag is
    omitted.
    """
    if arg and str(arg).startswith(
        ("ort_cpp:", "tensorrt:", "openvino:", "openvino_int8:")
    ):
        return arg
    return arg if arg is not None else existing_validation(default)


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
    # Path flags are optional: ``default=None`` defers resolution of the
    # documented defaults to ``run()`` via ``existing_validation``.

    parser.add_argument(
        "--model1", type=str, default=None,
        help=f"Reference model (default: {DEFAULT_MODEL_PT}). Prefix "
        f"'ort_cpp:' for the C++ backend (requires the built ort_cpp exe; "
        f"see docs/ORT_CPP.md), 'tensorrt:' to compare a serialized .engine "
        f"via the TensorRT backend, e.g. "
        f"'tensorrt:models/yolov8s_fp16.engine'. Prefix 'openvino:' or "
        f"'openvino_int8:' to compare an OpenVINO IR, e.g. "
        f"'openvino_int8:models/yolov8s_openvino_int8.xml' "
        f"(OPENVINO_DEVICE env var selects CPU/GPU/AUTO).",
    )
    parser.add_argument(
        "--model2", type=str, default=None,
        help=f"Comparison model (default: {DEFAULT_MODEL_FP32}). Accepts the "
        f"'ort_cpp:', 'tensorrt:', 'openvino:' and 'openvino_int8:' prefixes "
        f"like --model1.",
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
        "--device", type=str, default="cpu",
        help="Validation device: 'cpu', 'cuda', or a numeric GPU id "
             "(default: cpu). A numeric id selects the GPU the TensorRT "
             "engine runs on when a 'tensorrt:' model spec is used "
             "(e.g. --device 1); 'cpu'/'cuda' map to GPU 0 there. A numeric "
             "id affects the TensorRT engine only — the reference model, "
             "the sampler, and preprocessing then run on CPU; 'cuda' is "
             "the setting that runs the reference model on GPU.",
    )
    # Default 1e-3 / 1e-2 covers both PT↔ONNX-FP32 and FP32↔INT8: in detection
    # mode the tensor stats are advisory. For PT↔FP32 tensor runs pass
    # --atol 1e-4 --rtol 1e-3.
    parser.add_argument(
        "--atol", type=float, default=1e-3,
        help="Absolute tolerance (default: 1e-3 — works for both PT↔FP32 and FP32↔INT8)",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-2,
        help="Relative tolerance (default: 1e-2 — works for both PT↔FP32 and FP32↔INT8)",
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
    device = str(args.device)
    if device != "cpu" and device != "cuda" and not device.isdigit():
        raise ValueError(
            f"Invalid --device {args.device!r}: use 'cpu', 'cuda', or a "
            "numeric GPU id."
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
        model1=_resolve_consistency_model(args.model1, DEFAULT_MODEL_PT),
        model2=_resolve_consistency_model(args.model2, DEFAULT_MODEL_FP32),
        imgs_input=(args.imgs_input if args.imgs_input is not None
                    else existing_validation(DEFAULT_DATA_DIR)),
        mode=args.mode,
        max_images=args.max_images,
        img_sizes=args.img_sizes,
        batch_sizes=args.batch_sizes,
        resnet50=args.resnet50,
        device=args.device,
        atol=args.atol,
        rtol=args.rtol,
        report_path=report_path,
    )
    # Log the full model spec (not Path.name): for prefixed specs the
    # backend prefix is the only signal of which engine runs.
    logger.info(
        f"Starting consistency check: {cfg.model1} vs {cfg.model2}"
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
