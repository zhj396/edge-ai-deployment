"""Consistency validation between two YOLOv8s backends.

Two modes
---------
* **tensor** — strict ``np.allclose`` after PyTorch <-> ONNX FP32 export. Use case:
catching export bugs (graph rewrite errors, dtype drift).
* **detection** — looser IoU/class/score thresholds after ONNX FP32 vs INT8 quantization.
Use case: catching catastrophic accuracy loss.

Output: ``results/consistency_report.json`` plus per-image copies of failed samples under
``results/consistency_failed/``.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.dlpack import to_dlpack
from tqdm import tqdm
from ultralytics import YOLO

from utils import get_logger, select_providers
# Pure-NumPy comparison helpers live in utils.comparison so the test suite
# doesn't need ultralytics. Re-exported below for backward compatibility with
# ``from src.consistency import compare_tensors`` etc.
from utils.comparison import (  # noqa: F401  (re-exports — see comment above)
    cosine_similarity,
    _safe_pct,
    compare_tensors,
    compute_iou,
    _match_predictions_to_gt,
    compare_detections,
)
from . import post_process, preprocess_imgs

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Session / forward helpers
# ---------------------------------------------------------------------------
def get_ort_session(onnx_path: str, device: str) -> ort.InferenceSession:
    """Create an ORT ``InferenceSession`` with the project's shared provider selection so
    consistency runs on the *same* CUDA EP config as ``infer`` and the benchmark — not ORT's
    bare defaults."""
    providers = select_providers(device)
    logger.info("ONNX Runtime providers: %s (device=%s)", providers, device)

    session = ort.InferenceSession(onnx_path, providers=providers)
    active = session.get_providers()

    if device == "cuda" and "CUDAExecutionProvider" not in active:
        logger.warning("CUDAExecutionProvider requested but not active; fell back to %s", active)
    return session


def pt_forward(model: YOLO, x: torch.Tensor) -> np.ndarray:
    """Run a PyTorch YOLO forward pass and return numpy output."""
    with torch.no_grad():
        out = model.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        assert isinstance(out, torch.Tensor), (
            f"PyTorch output is not a Tensor: {type(out)}"
        )
        return out.detach().cpu().numpy()


def ort_forward(
    session: ort.InferenceSession, x: torch.Tensor, device: str
) -> np.ndarray:
    """Run an ORT forward pass and return numpy output (CUDA via DLPack)."""
    if device == "cuda" and torch.cuda.is_available():
        try:
            # ``to_dlpack`` raises on non-contiguous tensors; ``.contiguous()`` is a no-op when the
            # tensor is already contiguous (the common case from preprocess_imgs) and a cheap safety
            # net otherwise. Mirrors the CUDA branch of ``YOLOv8Engine._forward``.

            ort_input = ort.OrtValue.from_dlpack(to_dlpack(x.contiguous()))
        except Exception as e:
            logger.warning("DLPack conversion failed, falling back to CPU: %s", e)
            ort_input = x.detach().cpu().numpy().astype(np.float32)
    else:
        ort_input = x.detach().cpu().numpy().astype(np.float32)

    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: ort_input})[0]


# ---------------------------------------------------------------------------
# Unified model wrapper
# ---------------------------------------------------------------------------
class ModelWrapper:
    """Unified interface for PyTorch and ONNX models."""

    def __init__(self, model_path: str, device: str) -> None:
        self.path = model_path
        self.device = (
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        suffix = Path(model_path).suffix.lower()
        self.type = "pt" if suffix in (".pt", ".pth") else "onnx"
        self.model = self._load_model()

    def _load_model(self):
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Model file not found: {self.path}")

        try:
            if self.type == "pt":
                m = YOLO(self.path)
                m.to(self.device)
                m.model.eval()
                logger.info("PyTorch model loaded: %s", self.path)
                return m
            session = get_ort_session(self.path, self.device)
            logger.info("ONNX model loaded: %s", self.path)
            return session
        except Exception as e:
            logger.exception("Model load failed [%s]: %s", self.type, self.path)
            raise RuntimeError(f"Cannot load model {self.path}") from e

    def forward(self, x: torch.Tensor) -> np.ndarray:
        if self.type == "pt":
            return pt_forward(self.model, x)
        return ort_forward(self.model, x, self.device)


# ---------------------------------------------------------------------------
# Tensor & detection comparison helpers live in ``utils/comparison`` so the
# pure-Python unit tests don't need to load ultralytics. They are re-exported
# above (see ``from utils.comparison import ...``) for backward compatibility
# with ``from src.consistency import compare_tensors`` etc.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def make_json_serializable(obj):
    """Recursively coerce ``Path`` and friends to ``str``."""
    if isinstance(obj, (Path, os.PathLike)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    return obj


def atomic_json_dump(obj, path: Union[str, Path]) -> None:
    """Write JSON atomically via ``tmp + os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(obj), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def validate_consistency(
    model1: str,
    model2: str,
    imgs_input: Union[str, Path],
    mode: Literal["tensor", "detection"] = "tensor",
    max_images: int = 100,
    img_sizes: Optional[List[int]] = None,
    batch_sizes: Optional[List[int]] = None,
    resnet50: Optional[Path] = None,
    device: str = "cpu",
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    cosine_similarity_thresh: float = 0.995,
    mean_diff_thresh: float = 0.01,
    p99_thresh: float = 0.05,
    det_iou_thresh: float = 0.5,
    mean_iou_thresh: float = 0.92,
    class_match_thresh: float = 0.98,
    score_diff_thresh: float = 0.08,
    count_diff_thresh: int = 3,
    recall_match_thresh: float = 0.97,
    num_workers: int = 4,
    seed: int = 42,
    copy_failed_samples: bool = True,
    report_path: Union[str, Path] = "results/consistency_report.json",
) -> Dict:
    """Run consistency validation between two models.

    Use cases
    ---------
    1. PyTorch (.pt) vs ONNX FP32 — ``mode="tensor"``
    2. ONNX FP32 vs ONNX INT8 — ``mode="detection"``
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info("=" * 60)
    logger.info("YOLOv8s Consistency Validation")
    logger.info("Model1: %s", Path(model1).name)
    logger.info("Model2: %s", Path(model2).name)
    logger.info("Mode: %s | Device: %s", mode, device)

    model_obj1 = ModelWrapper(model1, device)
    model_obj2 = ModelWrapper(model2, device)

    data_yaml = Path(imgs_input) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # Imported lazily so that `from src.consistency import compare_tensors` (the
    # test path) doesn't pull in sampler → torchvision / imagehash / yaml.

    from .sampler import CalibrationSampler

    sampler = CalibrationSampler(
        data_yaml=data_yaml,
        calibration_size=max_images,
        local_weights=resnet50,
        device=device,
    )
    imgs = sampler.sample()
    if not imgs:
        raise RuntimeError("No valid images found")

    logger.info("Loaded %d images", len(imgs))

    img_sizes = img_sizes or [640]
    batch_sizes = batch_sizes or [1, 4, 8, 16]

    results: List[Dict] = []
    fail_dir = Path("results/consistency_failed")
    if copy_failed_samples:
        fail_dir.mkdir(parents=True, exist_ok=True)

    for imgsz in img_sizes:
        for bs in batch_sizes:
            logger.info("\nTest config: imgsz=%d, batch=%d", imgsz, bs)
            batch_results: List[Dict] = []
            failures = 0

            iterator = range(0, len(imgs), bs)
            for i in tqdm(iterator, desc=f"bs={bs}"):
                batch_paths = imgs[i : i + bs]
                try:
                    pre = preprocess_imgs(
                        batch_paths, imgsz=imgsz,
                        device=device, num_workers=num_workers,
                    )
                    x = pre["images"]

                    out1 = model_obj1.forward(x)
                    out2 = model_obj2.forward(x)

                    stats = compare_tensors(
                        out1, out2, atol, rtol,
                        cosine_similarity_thresh, mean_diff_thresh,
                        p99_thresh, mode,
                    )
                    stats.update({
                        "imgsz": imgsz,
                        "batch_size": len(batch_paths),
                        "mode": mode,
                    })

                    if mode == "detection":
                        dets1 = post_process(
                            out1, orig_shapes=pre["orig_shapes"],
                            conf_thres=conf_thres, iou_thres=iou_thres,
                            imgsz=imgsz,
                            ratios=pre.get("ratios"), pads=pre.get("pads"),
                        )
                        dets2 = post_process(
                            out2, orig_shapes=pre["orig_shapes"],
                            conf_thres=conf_thres, iou_thres=iou_thres,
                            imgsz=imgsz,
                            ratios=pre.get("ratios"), pads=pre.get("pads"),
                        )

                        per_image = []
                        for d1, d2, p in zip(dets1, dets2, pre["paths"]):
                            ds = compare_detections(
                                d1, d2, det_iou_thresh, mean_iou_thresh,
                                class_match_thresh, score_diff_thresh,
                                count_diff_thresh, recall_match_thresh,
                            )
                            ds["image_path"] = str(p)
                            per_image.append(ds)

                        if per_image:
                            det_summary = {
                                "det_passed": all(ds["passed"] for ds in per_image),
                                "det_mean_iou": float(np.mean(
                                    [ds["mean_iou"] for ds in per_image]
                                )),
                                "det_class_match_rate": float(np.mean(
                                    [ds["class_match_rate"] for ds in per_image]
                                )),
                                "det_score_diff_mean": float(np.mean(
                                    [ds["score_diff_mean"] for ds in per_image]
                                )),
                                "det_count_diff_mean": float(np.mean(
                                    [ds["count_diff"] for ds in per_image]
                                )),
                            }
                            stats.update(det_summary)
                            stats["per_image_detection"] = per_image

                            # In detection mode the per-image IoU/class/score gates are
                            # AUTHORITATIVE — they decide pass/fail. The tensor verdict
                            # (cosine/mean_diff/p99 on the raw 8400-box logit tensor) is kept
                            # only as an informational signal: INT8 routinely pushes raw-output
                            # p99 above 0.05 via noise in low-confidence boxes that NMS drops,
                            # so gating on it would fail runs whose final detections are
                            # identical. Catastrophic INT8 loss is still caught — it produces
                            # no/mismatched detections and fails the detection gates.

                            stats["tensor_gate_passed"] = bool(stats.get("passed", False))
                            stats["passed"] = bool(det_summary["det_passed"])
                            stats["status"] = "PASS" if stats["passed"] else "FAIL"

                            if not det_summary["det_passed"]:
                                logger.warning(
                                    "Detection comparison failed (batch_size=%d)",
                                    len(batch_paths),
                                )
                                for ds in (d for d in per_image if not d["passed"]):
                                    logger.warning(
                                        "%s | IoU=%.4f | Class=%.3f | "
                                        "ScoreDiff=%.4f | CountDiff=%d",
                                        Path(ds["image_path"]).name,
                                        ds["mean_iou"], ds["class_match_rate"],
                                        ds["score_diff_mean"], ds["count_diff"],
                                    )

                    batch_results.append(stats)

                    if not stats["passed"]:
                        failures += 1
                        logger.warning(
                            "Inconsistent | max_diff=%.6f | cos=%.6f",
                            stats.get("max_diff"),
                            stats.get("cosine_similarity"),
                        )
                        if copy_failed_samples:
                            for p in pre["paths"]:
                                logger.warning("Failed image: %s", p)
                                try:
                                    dst = fail_dir / f"{uuid.uuid4()}_{Path(p).name}"
                                    shutil.copy(p, dst)
                                except Exception:
                                    logger.exception("Failed to copy image")

                except Exception:
                    failures += 1
                    logger.exception("Batch processing failed")

            # Number of batches actually attempted this config — exception batches are counted in
            # `failures` but NOT appended to batch_results, so using len(batch_results) as the
            # denominator would under-count and could push fail_rate above 1.0 when every batch
            # raises.
            n_attempted = max(1, (len(imgs) + bs - 1) // bs) if imgs else 1

            summary: Dict = {
                "imgsz": imgsz,
                "batch_size": bs,
                "total_batches": n_attempted,
                "failures": failures,
                "fail_rate": failures / n_attempted,
                "overall_status": "PASS" if failures == 0 else "FAIL",
                "detailed_results": batch_results,
            }

            for k in ("max_diff", "mean_diff", "p95", "p99",
                      "std_diff", "cosine_similarity"):
                vals = [s.get(k) for s in batch_results if k in s]
                if vals:
                    summary[f"{k}_mean"] = float(np.mean(vals))

            results.append(summary)
            # Write incrementally so a crash mid-run keeps partial results.
            atomic_json_dump({"results": results, "mode": mode}, report_path)

            logger.info("fail_rate=%.2f%%", summary["fail_rate"] * 100)

    overall_pass = all(r["overall_status"] == "PASS" for r in results)
    final_result = {"results": results, "overall_pass": overall_pass, "mode": mode}

    logger.info("=" * 60)
    logger.info("Consistency validation %s", "PASS" if overall_pass else "FAIL")
    gc.collect()
    return final_result
