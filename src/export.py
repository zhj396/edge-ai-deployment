import numpy as np
import torch
import onnx
import onnxruntime as ort
import shutil
from pathlib import Path
from ultralytics import YOLO

from utils import get_logger, select_providers

logger = get_logger(__name__)


# =========================================================
# ONNX FP32 Model Validation
# =========================================================
def validate_onnx_model(onnx_path: Path, imgsz: int, device: str = "cpu"):
    """Validate an exported ONNX model for structural correctness and runtime feasibility."""
    logger.info("Validating ONNX model...")

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    # Best-effort shape validation: infer_shapes runs for its validation
    # side-effect (it raises on broken graphs); failures are non-fatal so
    # the runtime probe below still runs.

    try:
        onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        logger.warning("Shape inference skipped: %s", e)

    # Runtime test — probe session providers come from utils.select_providers
    # (same CUDA EP options as deployment).

    providers = select_providers(device)

    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception:
        logger.exception("ORT Session creation failed")
        raise

    actual_providers = session.get_providers()
    logger.info("ORT Providers: %s", actual_providers)

    # Dynamic shape handling: probe batch=1, and — when the export is
    # dynamic (any symbolic dim on axis 0) — also batch=3, exercising
    # the dynamic-batch path.
    def _probe(batch: int) -> None:
        inputs = {}
        for inp in session.get_inputs():
            shape = [batch if (isinstance(d, str) or d is None) else d
                     for d in inp.shape]
            if len(shape) == 4:
                shape[2], shape[3] = imgsz, imgsz
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
        session.run(None, inputs)

    _probe(1)
    first = session.get_inputs()[0]
    if isinstance(first.shape[0], str) or first.shape[0] is None:
        _probe(3)
        logger.info("Dynamic batch verified at batch=3")
    logger.info("ONNX validation passed")


# =========================================================
# ONNX FP32 Model Export
# =========================================================
def model_export(model_path: Path, output_path: Path, **kwargs):
    """Export YOLOv8 PyTorch model to ONNX FP32 format."""
    # Suffix gate: existing_validation accepts .xml/.bin for the openvino
    # command group; YOLO() consumes a .pt/.pth checkpoint, and a foreign
    # suffix fails inside ultralytics with an unactionable loader error.
    if Path(model_path).suffix.lower() not in (".pt", ".pth"):
        raise ValueError(
            f"export expects a .pt/.pth checkpoint, got: {model_path}. "
            f"OpenVINO IR files (.xml/.bin) belong to the `openvino` "
            f"command group."
        )
    model = YOLO(str(model_path))

    # device: "cpu"/"cuda" or an int GPU index; an int is passed through to
    # ultralytics' export. "cuda" -> GPU 0; else CPU.
    dev = kwargs.get("device", "cpu")
    if isinstance(dev, int) and not isinstance(dev, bool):
        device = (
            dev if torch.cuda.is_available() and dev < torch.cuda.device_count()
            else "cpu"
        )
    elif dev == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if device != dev:
        # A requested GPU must not silently fall back to CPU: the user would
        # believe the export ran on the GPU.
        logger.warning("Device %r unavailable (CUDA not ready) — falling back to %r", dev, device)
    logger.info("Exporting Model ON %s", device)

    exported = model.export(
        format="onnx",
        imgsz=kwargs.get("imgsz", 640),
        opset=kwargs.get("opset", 17),
        dynamic=kwargs.get("dynamic", True),
        simplify=kwargs.get("simplify", True),
        device=device,
        nms=kwargs.get("nms", False)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported_path = Path(exported)
    # Compare resolved paths so different spellings of the same file
    # (absolute vs relative) count as equal.
    if exported_path.resolve() != output_path.resolve():
        # The exporter writes next to the source .pt; move it to --output.
        # shutil.move also copies across drives.
        if output_path.exists():
            logger.warning("Overwriting existing %s", output_path)
        shutil.move(str(exported_path), str(output_path))

    if kwargs.get("validate", True):
        validate_onnx_model(output_path, kwargs.get("imgsz", 640), kwargs.get("device", "cpu"))
    logger.info(f"Export successful: {output_path}")
