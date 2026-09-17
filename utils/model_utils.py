"""Model utilities — metadata extraction, hash, file size.

Deployment pipelines need to:

* Verify model integrity via SHA-256.
* Read ONNX metadata (inputs, outputs, opset, producer, IR version).
* Compare two model files quickly.
* Report model size for storage / OTA planning.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Union

import onnx
import onnxruntime as ort

logger = logging.getLogger(__name__)


@dataclass
class OnnxMetadata:
    path: str
    file_size_mb: float
    sha256: str
    opset: int
    ir_version: int
    producer_name: str
    producer_version: str
    inputs: List[Dict]
    outputs: List[Dict]

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
def sha256_of_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of a file, streamed so it works on multi-GB models."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def file_size_mb(path: Union[str, Path]) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


# ---------------------------------------------------------------------------
def inspect_onnx(path: Union[str, Path]) -> OnnxMetadata:
    """Return a structured view of an ONNX file's metadata."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ONNX file not found: {p}")

    model = onnx.load(str(p))
    # Shape inference is best-effort validation: some INT8 QDQ graphs and externally-rewritten
    # models raise InferenceError even though the model is otherwise valid. Don't let that crash
    # `inspect` — the metadata below is read from the raw graph, not from the inferred shapes.

    try:
        onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        logger.warning("Shape inference skipped for %s: %s", p, e)

    def _io_info(value_info) -> Dict:
        shape = []
        for d in value_info.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else d.dim_param)
        dtype = value_info.type.tensor_type.elem_type
        return {
            "name": value_info.name,
            "shape": shape,
            "dtype": onnx.TensorProto.DataType.Name(dtype),
        }

    return OnnxMetadata(
        path=str(p),
        file_size_mb=round(file_size_mb(p), 3),
        sha256=sha256_of_file(p),
        opset=model.opset_import[0].version if model.opset_import else -1,
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        inputs=[_io_info(i) for i in model.graph.input],
        outputs=[_io_info(o) for o in model.graph.output],
    )


def select_providers(device: str) -> List:
    """ORT provider list, preferring a *tuned* CUDA EP when available.

    The single source of truth for provider selection across the project (engine, benchmark,
    consistency) so that every ORT session runs the same CUDA config and benchmark numbers reflect
    what ``infer`` actually deploys.

    The CUDA EP carries deployment-tuned options:

    * ``arena_extend_strategy=kNextPowerOfTwo`` — allocator rounds up to power-of-two pool
    buckets, reducing fragmentation (≤2× RAM cost).
    * ``cudnn_conv_algo_search=HEURISTIC`` — faster startup than ``EXHAUSTIVE`` (which benchmarks
    every algo on first call).
    * ``cudnn_conv_use_max_workspace=1`` — let cuDNN pick large-workspace conv kernels, which are
    usually the fastest.

    CPU is always included as a fallback so a CUDA-toolkit/driver mismatch degrades gracefully.
    Availability is checked via ``ort.get_available_providers()`` (not
    ``torch.cuda.is_available()``): ORT-gpu may be absent even when torch sees CUDA, and only ORT's
    own list tells us the CUDA EP is actually usable.
    """
    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "1",
                },
            ),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


def compare_models(path1: Union[str, Path], path2: Union[str, Path]) -> Dict:
    """Quick side-by-side comparison: size, SHA-256, opset.

    Both paths must point to valid ONNX files. Non-ONNX inputs raise
    ``onnx.checker.ValidationError`` / protobuf decode errors — caught here and
    surfaced as ``opset = -1`` so callers can still see the size and hash.
    """
    p1, p2 = Path(path1), Path(path2)
    opset_1, opset_2 = -1, -1
    try:
        opset_1 = inspect_onnx(p1).opset
    except Exception:
        pass
    try:
        opset_2 = inspect_onnx(p2).opset
    except Exception:
        pass
    return {
        "model_1": {
            "path": str(p1),
            "size_mb": round(file_size_mb(p1), 3),
            "sha256": sha256_of_file(p1),
            "opset": opset_1,
        },
        "model_2": {
            "path": str(p2),
            "size_mb": round(file_size_mb(p2), 3),
            "sha256": sha256_of_file(p2),
            "opset": opset_2,
        },
        "size_ratio": round(file_size_mb(p1) / max(file_size_mb(p2), 1e-9), 3),
        "identical": sha256_of_file(p1) == sha256_of_file(p2),
    }
