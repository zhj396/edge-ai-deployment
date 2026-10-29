"""CLI subcommand for the TensorRT backend — build / export / run.

Workflow
--------
1. ``python main.py tensorrt build  --model models/yolov8s.pt --precision fp16``
   .pt -> TRT-friendly ONNX -> TensorRT engine via the Python API (Path A,
   primary). For an existing ONNX, the ``*_trt`` export (opset 13,
   no-simplify) is the validated input config; the canonical
   ``yolov8s_fp32.onnx`` (opset 17 + simplify) carries the dynamic
   Shape+Slice subgraph named in the ``--model`` help — see docs/TENSORRT.md.
2. ``python main.py tensorrt export --model models/yolov8s.pt --precision fp16``
   .pt -> .engine via Ultralytics native export (Path B, Kaggle/Colab).
3. ``python main.py tensorrt run   --model models/yolov8s_fp16.engine --imgs-input data/images/val``
   Load the engine and run inference.

TensorRT is GPU-only and an optional dep — every subcommand guards on
TensorRT availability (``trt_build_available()`` for ``build``,
``tensorrt_available()`` for ``export`` / ``run``) and raises a
``SystemExit`` pointing at ``requirements-tensorrt.txt`` when the wheels
are missing.
"""
import json
from pathlib import Path

from utils import existing_validation, get_logger
from . import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_TENSORRT_FP16,
    DEFAULT_MODEL_TENSORRT_INT8,
    DEFAULT_MODEL_TRT_ONNX,
    DEFAULT_MODEL_TRT_ONNX_STATIC,
    TensorRTBuildConfig,
    TensorRTExportConfig,
    TensorRTRunConfig,
)
from src import (
    build_tensorrt_engine,
    export_tensorrt_ultralytics,
    tensorrt_available,
    trt_build_available,
)

logger = get_logger(__name__)


def _precision_default_output(precision: str) -> Path:
    """Precision-aware default .engine output name, shared by ``build`` /
    ``export``: an INT8 build lands at yolov8s_int8.engine; fp16 and fp32
    keep the _fp16 name."""
    return (Path(DEFAULT_MODEL_TENSORRT_INT8) if precision == "int8"
            else Path(DEFAULT_MODEL_TENSORRT_FP16))


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "tensorrt",
        help="TensorRT engine build / export / run (GPU, optional dep)",
    )
    sub = parser.add_subparsers(dest="trt_command", required=True)

    # ---- build (Path A: ONNX -> engine via Python API) ----
    p_build = sub.add_parser(
        "build", help="ONNX -> TensorRT engine via the Python API (Path A)",
    )
    p_build.add_argument("--model", type=existing_validation, required=True,
                         help="Input ONNX file, or a .pt (auto-exported to a "
                              "TRT-friendly ONNX: opset 13, no-simplify, "
                              "dynamic unless --static — the canonical "
                              "opset-17+simplify ONNX carries the subgraph "
                              "behind the Turing build-failure signature, "
                              "see docs/TENSORRT.md)")
    p_build.add_argument("--output", type=Path, default=None,
                         help="Output .engine path (default: "
                              "models/yolov8s_int8.engine for --precision int8, "
                              "models/yolov8s_fp16.engine otherwise — an fp32 "
                              "build keeps the _fp16 name)")
    p_build.add_argument("--precision", type=str,
                         choices=["fp32", "fp16", "int8"], default="fp16")
    p_build.add_argument("--imgsz", type=int, default=640)
    p_build.add_argument("--max-batch", type=int, default=8,
                         help="Max batch of the optimization profile (1..N)")
    p_build.add_argument("--workspace-gb", type=float, default=8.0,
                         help="Builder workspace pool (GiB)")
    p_build.add_argument(
        "--calib-imgs-input", default=None,
        help="Calibration dataset dir (needs data.yaml; default: data) — "
             "INT8 only. The shared CalibrationSampler builds the image "
             "list from it.",
    )
    p_build.add_argument("--calib-cache", type=Path,
                         default="models/calibration.cache",
                         help="Calibration cache path (read/written; "
                              "--calibrator path only — the default QDQ "
                              "path and fp16/fp32 builds ignore it)")
    p_build.add_argument("--max-cal-samples", type=int, default=300)
    p_build.add_argument(
        "--calib-method", type=str, choices=["MinMax", "Entropy"],
        default="MinMax",
        help="INT8 QDQ calibration method (MinMax default). Entropy (KL) may "
             "fit tight activation distributions better under the forced "
             "symmetric quantization TRT requires (see docs/TENSORRT.md). "
             "QDQ-path only (no-op under --calibrator). NOTE: Entropy holds "
             "all intermediate tensors in host memory during calibration "
             "(ORT's HistogramCalibrater); reduce --max-cal-samples "
             "substantially when using it (e.g. to a few dozen).",
    )
    p_build.add_argument("--resnet50", type=Path,
                         default="models/resnet50-11ad3fa6.pth")
    p_build.add_argument("--device", type=int, default=0,
                         help="GPU id (CUDA_VISIBLE_DEVICES); also selects "
                              "the GPU the calibration feature extraction "
                              "runs on")
    p_build.add_argument(
        "--static", action="store_true",
        help="Build a static-batch=1 engine from a static-batch ONNX (no "
             "optimization profile). The Turing (sm_75, e.g. T4) route: the "
             "dynamic-batch DFL reshape forces a Shape+Slice subgraph that "
             "TRT 10.x failed to lower in the author's T4 session (observed "
             "signature: 'nbDims > Dims::MAX_DIMS'; cause inferred, not "
             "documented; pinned-environment verdict _TBD_ — "
             "docs/TENSORRT.md). On Ampere+ (sm_80+) the dynamic route "
             "(--static off, 1..max_batch) is the expected default.",
    )
    p_build.add_argument(
        "--no-exclude-head", dest="exclude_head", action="store_false",
        default=True,
        help="Pin the /model.22 Detect head to FP32 under INT8 (default ON — "
             "keeps the head's box-decode accumulation in FP32). Pass "
             "--no-exclude-head to quantize the whole net (ablation only). "
             "No-op for FP16/FP32 builds. Calibrator-path (--calibrator) "
             "only; the default QDQ path excludes the head via "
             "quantize_onnx_to_int8 regardless.",
    )
    p_build.add_argument(
        "--calibrator", dest="calibrator", action="store_true", default=False,
        help="INT8 ablation: use the calibrator+OBEY path (plain ONNX + "
             "IInt8EntropyCalibrator2 + layer.precision/OBEY_PRECISION_"
             "CONSTRAINTS); the default is the QDQ path. NOTE: under "
             "calibrator-based implicit INT8 the precision constraints do not "
             "bind at tactic selection, so the /model.22/ head may run INT8; "
             "the default QDQ path (drop --calibrator) holds the head FP32 by "
             "QDQ omission.",
    )

    # ---- export (Path B: .pt -> engine via Ultralytics) ----
    p_exp = sub.add_parser(
        "export", help=".pt -> .engine via Ultralytics native export (Path B)",
    )
    p_exp.add_argument("--model", type=existing_validation, required=True,
                       help="Input PyTorch .pt file")
    p_exp.add_argument("--output", type=Path, default=None,
                       help="Output .engine path (default: "
                            "models/yolov8s_int8.engine for --precision int8, "
                            "models/yolov8s_fp16.engine otherwise — an fp32 "
                            "export keeps the _fp16 name)")
    p_exp.add_argument("--precision", type=str,
                       choices=["fp32", "fp16", "int8"], default="fp16")
    p_exp.add_argument("--imgsz", type=int, default=640)
    p_exp.add_argument(
        "--data", type=Path, default=Path(DEFAULT_DATA_DIR) / "data.yaml",
        help="data.yaml for INT8 calibration (required for --precision int8)",
    )
    p_exp.add_argument("--device", type=int, default=0)

    # ---- run ----
    p_run = sub.add_parser("run", help="Run TensorRT inference")
    p_run.add_argument("--model", type=existing_validation, required=True,
                       help="Path to the .engine")
    p_run.add_argument("--imgs-input", type=existing_validation, required=True)
    p_run.add_argument("--imgsz", type=int, default=640)
    p_run.add_argument("--device", type=int, default=0)
    p_run.add_argument("--max-batch", type=int, default=8,
                       help="Max batch of the profile the engine was built "
                            "with; forwards are capped at this (sub-looped).")
    p_run.add_argument("--max-imgs", type=int, default=32)
    p_run.add_argument("--batch-size", type=int, default=8)
    p_run.add_argument("--conf", type=float, default=0.25)
    p_run.add_argument("--iou", type=float, default=0.45)
    p_run.add_argument(
        "--data", type=Path, default=Path(DEFAULT_DATA_DIR) / "data.yaml",
        help="data.yaml carrying the class-name list. A serialized TRT engine "
             "has no ultralytics metadata, so without this the annotated "
             "images carry 'class_N' placeholder names. Ignored if absent.",
    )
    p_run.add_argument(
        "--output-dir", type=Path,
        default="results/predictions/tensorrt")
    p_run.add_argument(
        "--backend", type=str, choices=["python"], default="python",
        help="Which TensorRT engine to drive: 'python' (cuda-python, the "
             "default — mirrors `tensorrt:`).",
    )

    return parser


def run(args):
    if args.trt_command == "build":
        _run_build(args)
    elif args.trt_command == "export":
        _run_export(args)
    elif args.trt_command == "run":
        _run_inference(args)
    else:
        raise SystemExit(f"Unknown tensorrt subcommand: {args.trt_command}")


def _run_build(args):
    if not trt_build_available():
        raise SystemExit(
            "tensorrt is not installed. Run: "
            "pip install -r requirements-tensorrt.txt (on top of "
            "requirements-kaggle.txt)."
        )
    is_int8 = args.precision == "int8"
    if args.calibrator and not is_int8:
        logger.warning(
            "--calibrator is an INT8 ablation; it is ignored for "
            "precision=%s.", args.precision,
        )
    # Precision-aware default output path: an INT8 build lands at
    # yolov8s_int8.engine (the same resolution `export` uses).
    if args.output is not None:
        output = args.output
    else:
        output = _precision_default_output(args.precision)
    calib_images = None
    # INT8-only calibration inputs, resolved at run() time so a fresh clone
    # without data/ can still build fp16/fp32 engines.
    calib_imgs_input = None
    resnet50 = None
    if is_int8:
        calib_imgs_input = existing_validation(
            args.calib_imgs_input or DEFAULT_DATA_DIR
        )
        if Path(args.resnet50).exists():
            resnet50 = Path(args.resnet50)
        else:
            logger.warning(
                "%s does not exist; the sampler will download torchvision's "
                "pretrained weights (fails on an offline host).",
                args.resnet50,
            )
    cfg = TensorRTBuildConfig(
        model=args.model,
        output=output,
        imgsz=args.imgsz,
        precision=args.precision,
        max_batch=args.max_batch,
        workspace_bytes=int(args.workspace_gb * 1024 ** 3),
        device=args.device,
        static=args.static,
        exclude_head=args.exclude_head,
        calibrator=args.calibrator,
        calib_imgs_input=calib_imgs_input,
        calib_cache=args.calib_cache if is_int8 else None,
        max_cal_samples=args.max_cal_samples,
        resnet50=resnet50,
        calib_method=args.calib_method,
    )

    # Resolve the ONNX to build from. A .pt is exported to a TRT-friendly ONNX
    # first: the canonical opset-17 + onnxsim-simplify ONNX carries a dynamic
    # Shape+Slice subgraph behind the failure signature the author's Turing
    # T4 session produced (observed "nbDims > Dims::MAX_DIMS"; cause inferred
    # from the log, not documented; pinned-environment verdict _TBD_ —
    # docs/TENSORRT.md). The TRT-friendly export (opset 13, no simplify,
    # dynamic unless --static) is the config validated on the T4.
    # A .onnx is used as-is — pass a TRT-friendly one (or just
    # --model X.pt, which exports it).
    onnx_path = Path(cfg.model)
    if onnx_path.suffix.lower() in (".pt", ".pth"):
        from src import model_export
        # A static build exports to its own ONNX name so it cannot collide
        # with the dynamic export cached at DEFAULT_MODEL_TRT_ONNX (the same
        # split scripts/build_trt_engines.sh uses for its cache).
        onnx_path = (
            Path(DEFAULT_MODEL_TRT_ONNX_STATIC)
            if cfg.static else Path(DEFAULT_MODEL_TRT_ONNX)
        )
        logger.info(
            "Exporting TRT-friendly ONNX (%s, opset 13, no-simplify, "
            "dynamic=%s) -> %s", onnx_path.name, not cfg.static, onnx_path,
        )
        model_export(
            model_path=cfg.model,
            output_path=onnx_path,
            imgsz=cfg.imgsz,
            opset=13,
            dynamic=not cfg.static,
            simplify=False,
            device=cfg.device,
            nms=False,
            validate=False,
        )
    else:
        if onnx_path.suffix.lower() != ".onnx":
            raise SystemExit(
                f"--model must be a .pt/.pth or .onnx file, got {cfg.model}; "
                "a serialized .engine is the output of this command, not an "
                "input"
            )
        if cfg.static:
            logger.warning(
                "--static set but --model is a .onnx; ensure it was exported "
                "with --no-dynamic (static batch=1), else the build fails on "
                "the non-1 batch dim. Pass --model X.pt to auto-export static."
            )
        if not onnx_path.stem.endswith(("_trt", "_trt_static", "_trt_int8")):
            # The TRT-friendly exports this command (or a previous run)
            # produced carry the _trt / _trt_static / _trt_int8 stems —
            # none of them carries the canonical opset-17 +simplify
            # subgraph, so no warning is needed for those.
            logger.warning(
                "Building from %s directly. If it is the canonical opset-17 "
                "+simplify ONNX, TRT 10.x may hit the Turing build failure "
                "(observed signature 'nbDims > Dims::MAX_DIMS'); pass "
                "--model X.pt instead to auto-export a TRT-friendly ONNX.",
                onnx_path,
            )

    if is_int8:
        # INT8 needs calibration images either way: the default QDQ path feeds
        # them to ORT's quantize_onnx_to_int8 (which builds its own sampler);
        # the --calibrator ablation feeds the list to TRT's
        # IInt8EntropyCalibrator2.
        data_yaml = Path(cfg.calib_imgs_input) / "data.yaml"
        if not data_yaml.exists():
            raise SystemExit(f"data.yaml not found in {cfg.calib_imgs_input}")

        if cfg.calibrator:
            # Calibrator ablation: build the image list for TRT's calibrator.
            # Under calibrator-based implicit INT8 the precision constraints
            # may not bind at tactic selection, so the head may still run
            # INT8 — see docs/TENSORRT.md.
            from src import CalibrationSampler
            sampler = CalibrationSampler(
                data_yaml=data_yaml,
                calibration_size=cfg.max_cal_samples,
                local_weights=cfg.resnet50,
                device=cfg.device,
            )
            calib_images = [str(p) for p in sampler.sample()]
            logger.info("INT8 (calibrator ablation) calibration images: %d",
                        len(calib_images))
        else:
            # Default QDQ path: run quantize_onnx_to_int8 (the /model.22/
            # head-exclusion scope) on the TRT-friendly ONNX. Produces a
            # head-excluded QDQ ONNX that TRT consumes with the INT8 flag and
            # NO calibrator -- the head stays FP32 by QDQ omission. An input
            # that already carries QDQ nodes is passed through as-is:
            # re-quantizing it would double-quantize the graph.
            from src.tensorrt_build import _is_qdq_onnx
            if _is_qdq_onnx(str(onnx_path)):
                qdq_path = onnx_path
                logger.info(
                    "%s already carries QDQ nodes; skipping re-quantization "
                    "(TRT consumes it with the INT8 flag, no calibrator).",
                    onnx_path.name,
                )
            else:
                qdq_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
                logger.info(
                    "QDQ quantization (quantize_onnx_to_int8, "
                    "/model.22/ head excluded) -> %s | method=%s",
                    qdq_path, cfg.calib_method,
                )
                from src import quantize_onnx_to_int8
                from onnxruntime.quantization import QuantType
                # TRT-compatible QDQ: TRT 10.x requires FULLY SYMMETRIC quantization
                # (every QuantizeLinear/DequantizeLinear zero-point must be all
                # zeros) — it rejects asymmetric activations ("only supports
                # symmetric quantization"). So: ActivationSymmetric=True (forces
                # activation zero-point=0; WeightSymmetric is already True in the
                # base options) + activation_type=QInt8 (signed, no UINT8 zero-
                # point) + QuantizeBias=False (Conv bias left FP32 — TRT 10.x's
                # DequantizeLayer rejects the INT32 bias DQ ORT emits by default,
                # "can only run in kINT8/kFP8/kINT4"; TRT folds FP32 bias natively).
                # The head (/model.22/) is excluded -> stays FP32 by QDQ omission.
                # calib_method comes from --calib-method (MinMax default; Entropy
                # may tighten the symmetric activation scales, but holds all
                # intermediate tensors in host memory — see --max-cal-samples
                # help).
                # device=cfg.device: cfg.device is the int GPU id (the same value
                # build_tensorrt_engine uses for CUDA_VISIBLE_DEVICES / cuDeviceGet).
                # CalibrationSampler maps the int to "cuda:<id>", and
                # select_providers parses "cuda:<id>" back to device_id=<id>, so
                # the QDQ calibration forward runs on that GPU. The ORT session is
                # released by quantize_static before build_tensorrt_engine retains
                # the CUDA primary context (sequential, refcounted primary ctx).
                # Multi-GPU caveat (see build_tensorrt_engine): the
                # CUDA_VISIBLE_DEVICES pin only takes effect if no CUDA context
                # was initialized in this process yet, and the ORT CUDA-EP
                # calibration session initializes the driver — on multi-GPU
                # hosts, run each build in a fresh process for the pin to be
                # honored.
                trt_extra = {"QuantizeBias": False, "ActivationSymmetric": True}
                qdq_path = Path(quantize_onnx_to_int8(
                    onnx_fp32_path=str(onnx_path),
                    onnx_int8_path=str(qdq_path),
                    calibration_imgs=str(cfg.calib_imgs_input),
                    imgsz=cfg.imgsz,
                    max_samples=cfg.max_cal_samples,
                    resnet50=cfg.resnet50,
                    method=cfg.calib_method,
                    device=cfg.device,
                    activation_type=QuantType.QInt8,
                    extra_options=trt_extra,
                ))
                onnx_path = qdq_path
                logger.info(
                    "QDQ ONNX ready (%s); TRT will build with INT8 flag, no "
                    "calibrator (head FP32 via QDQ omission).", qdq_path.name,
                )

    out = build_tensorrt_engine(
        onnx_path=str(onnx_path),
        output_path=str(cfg.output),
        imgsz=cfg.imgsz,
        fp16=cfg.precision == "fp16",
        int8=is_int8,
        max_batch=cfg.max_batch,
        workspace_bytes=cfg.workspace_bytes,
        calib_images=calib_images,
        calib_cache_path=str(cfg.calib_cache) if cfg.calib_cache else None,
        device=cfg.device,
        static=cfg.static,
        exclude_head=cfg.exclude_head,
    )
    logger.info("Engine build complete: %s", out)


def _run_export(args):
    if not tensorrt_available():
        raise SystemExit(
            "tensorrt is not installed. Run: "
            "pip install -r requirements-tensorrt.txt (on top of "
            "requirements-kaggle.txt)."
        )
    if Path(args.model).suffix.lower() != ".pt":
        raise SystemExit(
            f"--model must be a .pt file for the Ultralytics export, got "
            f"{args.model}; `tensorrt build` accepts an ONNX input"
        )
    data_yaml = args.data if (args.precision == "int8" and Path(args.data).exists()) else None
    if args.precision == "int8" and data_yaml is None:
        raise SystemExit("INT8 export requires --data pointing to a data.yaml")
    if args.output is not None:
        output = args.output
    else:
        # Precision-aware default output path (shared with `build`): an INT8
        # export lands at yolov8s_int8.engine.
        output = _precision_default_output(args.precision)
    cfg = TensorRTExportConfig(
        model=args.model,
        output=output,
        imgsz=args.imgsz,
        precision=args.precision,
        device=args.device,
        data_yaml=data_yaml,
    )
    out = export_tensorrt_ultralytics(
        pt_path=str(cfg.model),
        engine_path=str(cfg.output),
        imgsz=cfg.imgsz,
        precision=cfg.precision,
        data_yaml=str(cfg.data_yaml) if cfg.data_yaml else None,
        device=cfg.device,
    )
    logger.info("Ultralytics TRT export complete: %s", out)


def _run_inference(args):
    cfg = TensorRTRunConfig(
        model=args.model,
        imgs_input=args.imgs_input,
        imgsz=args.imgsz,
        device=args.device,
        max_imgs=args.max_imgs,
        batch_size=args.batch_size,
        conf=args.conf,
        iou=args.iou,
        data_yaml=args.data,
        output_dir=args.output_dir,
        max_batch=args.max_batch,
    )
    # Select the engine implementation. 'python' (the only choice) drives the
    # .engine via cuda-python.
    if not tensorrt_available():
        raise SystemExit(
            "tensorrt is not installed. Run: "
            "pip install -r requirements-tensorrt.txt (on top of "
            "requirements-kaggle.txt)."
        )
    from src import TensorRTEngine
    engine_cls = TensorRTEngine

    # Only hand the engine a data_yaml that actually exists; otherwise let it
    # fall back to its class_N default (mirrors cli/openvino.py::_run_inference).
    data_yaml = (
        cfg.data_yaml
        if cfg.data_yaml is not None and Path(cfg.data_yaml).exists()
        else None
    )
    engine = engine_cls(
        model_path=str(cfg.model),
        device=cfg.device,
        imgsz=cfg.imgsz,
        max_batch=cfg.max_batch,
        data_yaml=data_yaml,
    )
    try:
        detections = engine.infer(
            imgs_input=cfg.imgs_input,
            conf=cfg.conf,
            iou=cfg.iou,
            max_imgs=cfg.max_imgs,
            batch_size=cfg.batch_size,
            output_dir=cfg.output_dir,
        )
        for d in detections:
            logger.info(json.dumps(d, ensure_ascii=False))
        logger.info("TensorRT inference complete (%s backend)", args.backend)
    finally:
        engine.release()
