from pathlib import Path

from utils import get_logger
from . import DEFAULT_MODEL_FP32, InferConfig, resolve_path_arg
from src import YOLOv8Engine

logger = get_logger(__name__)


# =========================================================
# Add CLI subparser
# =========================================================
def add_parser(subparsers):
    parser = subparsers.add_parser(
        "infer",
        help="Run model inference on images",
        epilog=(
            "Example: python main.py infer --backend onnx_int8 "
            "--model models/yolov8s_int8.onnx --imgs-input data/images/val "
            "--output-dir results/predictions/onnx_int8"
        ),
    )
    # ``default=None`` so argparse skips ``existing_validation`` on the default; the
    # documented default is resolved inside ``run()`` via :func:`resolve_model_arg`.
    parser.add_argument(
        "--model", type=Path, default=None,
        help=f"Model path (.pt/.onnx per --backend; default: {DEFAULT_MODEL_FP32})",
    )
    parser.add_argument(
        "--backend", choices=["pytorch", "onnx_fp32", "onnx_int8"], default="onnx_fp32",
    )
    parser.add_argument(
        "--imgs-input", type=Path, required=True,
        help="Image directory or single image path",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-imgs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--no-save", action="store_true",
        help="If specified, do not save annotated overlay images to --output-dir",
    )
    parser.add_argument("--output-dir", type=Path, default="results/predictions")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Inference device (default: cpu)",
    )

    return parser


# =========================================================
# Run inference command
# =========================================================
def run(args):
    cfg = InferConfig(
        model=resolve_path_arg(args.model, DEFAULT_MODEL_FP32),
        backend=args.backend,
        imgs_input=args.imgs_input,
        imgsz=args.imgsz,
        max_imgs=args.max_imgs,
        batch_size=args.batch_size,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save=not args.no_save,
        output_dir=args.output_dir,
        device=args.device,
    )
    engine = YOLOv8Engine(
        model_path=cfg.model,
        backend=cfg.backend,
        imgsz=cfg.imgsz,
        device=cfg.device,
    )
    logger.info(f"Starting inference on: {cfg.imgs_input}")
    detections = engine.infer(
        imgs_input=cfg.imgs_input,
        conf=cfg.conf,
        iou=cfg.iou,
        max_imgs=cfg.max_imgs,
        batch_size=cfg.batch_size,
        max_det=cfg.max_det,
        save=cfg.save,
        output_dir=cfg.output_dir,
    )

    logger.info("Inference complete: %d detections", sum(len(d) for d in detections))
