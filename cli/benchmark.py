from typing import Dict
from pathlib import Path

from utils import get_logger
from . import BenchmarkConfig, DEFAULT_DATA_DIR, DEFAULT_MODEL_FP32, resolve_path_arg
from src import Benchmark

logger = get_logger(__name__)


# =========================================================
# Add CLI subparser
# =========================================================
def add_parser(subparsers):
    parser = subparsers.add_parser(
        "benchmark",
        help="Run performance benchmark",
        epilog=(
            "Example: python main.py benchmark --model pytorch:models/yolov8s.pt "
            "--model onnx_fp32:models/yolov8s_fp32.onnx "
            "--model onnx_int8:models/yolov8s_int8.onnx --imgs-input data"
        ),
    )
    parser.add_argument(
        "--model", action="append", metavar="BACKEND:PATH",
        help=(
            "Model config in 'backend:path' format. Can be specified multiple "
            "times. Example: --model pytorch:models/yolov8s.pt "
            "--model onnx_fp32:models/yolov8s_fp32.onnx "
            "--model onnx_int8:models/yolov8s_int8.onnx"
        ),
    )
    parser.add_argument(
        "--imgs-input", type=Path, default=None,
        help=f"Image directory containing data.yaml (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--conf-threshold", type=float, default=0.001,
        help="Confidence threshold for mAP validation (default: 0.001, COCO standard)",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.7,
                        help="NMS IoU threshold for mAP validation (default: 0.7)")
    parser.add_argument("--speed-conf", type=float, default=0.25,
                        help="Confidence threshold for the NMS inside the speed loop "
                        "(default: 0.25, the engine.infer deploy default). Lower it to "
                        "0.001 only if you want NMS-bound timing.")
    parser.add_argument("--speed-iou", type=float, default=0.45,
                        help="NMS IoU threshold for the speed loop (default: 0.45)")
    parser.add_argument(
        "--resnet50", type=Path, default=None,
        help="Optional local ResNet50 checkpoint (default: torchvision auto-download)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Benchmark device (default: cpu)",
    )
    parser.add_argument("--validation", action="store_true", help="Run validation (mAP)")
    parser.add_argument("--no-sampler", action="store_true",
                        help="Skip the CalibrationSampler (which loads ResNet-50) for the "
                        "speed-test image set and use sorted val paths instead. Use on "
                        "offline/CI boxes without torchvision, or for a quick speed check.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=25)

    return parser


# =========================================================
# Run benchmark command
# =========================================================
def run(args):
    model_dict: Dict[str, str] = {}
    if args.model:
        for item in args.model:
            backend, sep, path = item.partition(":")
            if not sep:
                raise ValueError(f"Invalid model format. Expected 'backend:path', got: {item}")

            backend = backend.strip()
            path = path.strip()

            if backend not in ["pytorch", "onnx_fp32", "onnx_int8"]:
                # A bare Windows path ("C:\models\x.onnx") partitions into
                # backend="C" — point the user at the real problem instead.
                if len(backend) == 1 and path.startswith(("\\", "/")):
                    raise ValueError(
                        f"Invalid model format. Expected 'backend:path', got: {item}"
                    )
                raise ValueError(f"Unsupported backend: {backend}")

            model_dict[backend] = path

    if not model_dict:
        # No --model given: fall back to the documented default FP32 ONNX, resolved (and
        # existence-validated) through the same single source of truth as every other subcommand.
        model_dict["onnx_fp32"] = str(resolve_path_arg(None, DEFAULT_MODEL_FP32))

    cfg = BenchmarkConfig(
        model=model_dict,
        imgs_input=resolve_path_arg(args.imgs_input, DEFAULT_DATA_DIR),
        imgsz=args.imgsz,
        max_images=args.max_images,
        batch_size=args.batch_size,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        speed_conf=args.speed_conf,
        speed_iou=args.speed_iou,
        validation=args.validation,
        resnet50=args.resnet50,
        device=args.device,
        warmup=args.warmup,
        runs=args.runs,
        use_sampler=not args.no_sampler,
    )

    benchmark = Benchmark(
        imgs_input=cfg.imgs_input,
        max_images_for_speed=cfg.max_images,
        imgsz=cfg.imgsz,
        batch_size=cfg.batch_size,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        speed_conf=cfg.speed_conf,
        speed_iou=cfg.speed_iou,
        validation=cfg.validation,
        resnet50=cfg.resnet50,
        device=cfg.device,
        warmup=cfg.warmup,
        runs=cfg.runs,
        use_sampler=cfg.use_sampler,
    )

    logger.info("========== Starting Benchmark ==========")
    summary = benchmark.run_all(cfg.model)
    logger.info("Benchmark complete: %d backends", len(summary))
