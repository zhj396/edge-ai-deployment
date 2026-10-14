"""Performance benchmark for YOLOv8s backends.

The benchmark measures end-to-end inference latency, throughput (FPS), and peak memory across
the supported backends (PyTorch, ONNX Runtime FP32 / INT8), plus optional mAP validation through
the backend-agnostic COCO evaluator (``utils/map_eval``).

Measurement notes
-----------------
* **Percentile latencies** (p50/p90/p95/p99) are reported alongside the mean.
* Memory tracking is backend-aware: CUDA rows report the torch allocator peak
(``torch.cuda.max_memory_allocated``); CPU rows report the process RSS sampled
once after the timed loop via ``psutil`` (RSS can shrink, so it is a lower
bound on peak RSS).
* Each backend run starts from a known memory baseline so the RSS-increase number reflects only that
backend's footprint.
* ``malloc_trim`` is invoked on Linux between runs to return freed pages to the OS, keeping the
RSS baseline stable across backends.
"""
from __future__ import annotations

import ctypes
import gc
import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil
import torch
from torch.utils.dlpack import to_dlpack
from ultralytics import YOLO

from utils import get_logger, percentiles, select_providers
from . import post_process, preprocess_imgs

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _thread_knobs() -> tuple:
    """The (intra, inter) thread values ``_make_session_options`` bakes in."""
    ncpu = os.cpu_count() or 1
    return min(4, ncpu), (2 if min(4, ncpu) > 1 else 1)


def _make_session_options() -> ort.SessionOptions:
    """ORT session options: single-stream thread config (intra=4, inter=2)."""
    intra, inter = _thread_knobs()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = intra
    opts.inter_op_num_threads = inter
    return opts


def _try_malloc_trim() -> None:
    """Ask glibc to return freed heap pages to the OS (Linux only)."""
    if platform.system() != "Linux":
        return
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
class Benchmark:
    """Latency / throughput / memory / mAP benchmark for YOLOv8s."""

    def __init__(
        self,
        imgs_input: Union[str, Path],
        max_images_for_speed: int = 16,
        imgsz: int = 640,
        batch_size: int = 1,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,  # COCO mAP-standard (matches --iou-threshold CLI default)
        speed_conf: float = 0.25,
        speed_iou: float = 0.45,
        validation: bool = False,
        resnet50: Optional[Path] = None,
        device: str = "cpu",
        warmup: int = 10,
        runs: int = 25,
        use_sampler: bool = True,
        timestamp_suffix: Optional[str] = None,
    ) -> None:
        self.imgs_input = imgs_input
        self.max_images_for_speed = max_images_for_speed
        self.imgsz = imgsz
        self.batch_size = batch_size
        # conf/iou_threshold drive the optional mAP *validation* (the native
        # COCO evaluator, utils/map_eval), which needs COCO-standard conf->0 /
        # iou=0.7 for a correct PR curve. speed_conf/speed_iou drive the NMS
        # inside the *speed* loop and default to engine.infer's deploy values
        # (0.25 / 0.45) so the timed region measures the deployed pipeline; at
        # conf=0.001 the NMS processes all 8400 candidate boxes and dominates
        # the timed region.

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.speed_conf = speed_conf
        self.speed_iou = speed_iou
        self.validation = validation
        self.resnet = resnet50
        self.device = (
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.warmup = warmup
        self.runs = runs
        self.use_sampler = use_sampler
        # Timestamp suffix appended to ``benchmark_summary.csv`` + every
        # ``<backend>_perclass.csv`` so each benchmark run leaves a persistent,
        # sortable artifact. The CLI generates it once per run; library callers
        # that omit it get the unsuffixed fixed path.
        self.timestamp_suffix = timestamp_suffix or ""

        # Speed-test image set: sample a diverse subset (sampler-selected) so
        # timings cover varied scenes.

        data_yaml = Path(imgs_input) / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
        if use_sampler:
            # CalibrationSampler gives a stratified + phash-dedup + farthest-first representative
            # subset, but it loads ResNet-50 (and may download weights). If it fails — e.g.
            # torchvision missing, weights unreachable on an offline/CI box —
            # fall back to the cheap path-only selector so the benchmark still runs.
            try:
                from . import CalibrationSampler
                sampler = CalibrationSampler(
                    data_yaml=data_yaml,
                    calibration_size=max_images_for_speed,
                    local_weights=self.resnet,
                )
                self.speed_imgs = sampler.sample()
            except Exception as e:
                logger.warning(
                    "CalibrationSampler unavailable (%s); using sorted val "
                    "paths instead (less representative, no sampling cost).", e,
                )
                self.speed_imgs = self._load_val_paths(
                    data_yaml, max_images_for_speed
                )
        else:
            # ``use_sampler=False``: skip ResNet-50 sampling entirely.

            self.speed_imgs = self._load_val_paths(
                data_yaml, max_images_for_speed
            )
        logger.info("Speed test set: %d images", len(self.speed_imgs))

    @property
    def summary_csv_path(self) -> str:
        """Canonical summary CSV path for this run."""
        return f"results/benchmark_summary{self.timestamp_suffix}.csv"

    @staticmethod
    def _load_val_paths(data_yaml: Path, n: int) -> List[Path]:
        """Cheap path-only speed-test set: read ``data.yaml``'s ``val`` dir and
        return the first ``n`` sorted image paths.

        No label parsing, no ResNet-50 — used when CalibrationSampler is disabled
        (``use_sampler=False``) or when it raises.
        """
        import yaml

        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        image_dir = (data_yaml.parent / cfg["val"]).resolve()
        if not image_dir.exists():
            raise FileNotFoundError(f"Val image dir not found: {image_dir}")
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        paths = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in suffixes
        )
        return paths[:n]

    # memory
    def _baseline_memory(self) -> Dict:
        """Snapshot RSS (and CUDA peak) before backend load."""
        gc.collect()
        _try_malloc_trim()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        return {
            "system_rss_mb": psutil.Process().memory_info().rss / (1024 ** 2),
        }

    def _peak_memory(self, baseline: Dict) -> Dict:
        info = {
            "system_rss_mb": psutil.Process().memory_info().rss / (1024 ** 2),
            "rss_increase_mb": 0.0,
        }
        info["rss_increase_mb"] = round(info["system_rss_mb"] - baseline["system_rss_mb"], 2)

        if self.device == "cuda":
            info["cuda_peak_mb"] = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
            )
        return info

    # batching
    @staticmethod
    def _build_batches(imgs: List, batch_size: int):
        for i in range(0, len(imgs), batch_size):
            yield imgs[i : i + batch_size]

    # mAP (native, backend-agnostic)
    def _compute_map_native(
        self, backend: str, model_path: str
    ) -> Tuple[Optional[Dict], str]:
        """Run the backend-agnostic COCO mAP evaluator on ``backend``.

        Constructs the engine matching ``backend`` and drives its shared
        ``infer()`` API through ``utils.map_eval.evaluate_map`` (conf=0.001,
        iou=0.7 — the COCO mAP-standard NMS config, from
        ``self.conf_threshold`` / ``self.iou_threshold``). Every CLI backend
        goes through this evaluator, so their mAP50 / mAP50-95 share one COCO
        metric and are directly comparable across backends.

        Returns ``(result, mAP_source)``: ``result`` is
        ``{"mAP50", "mAP50-95"}`` (values None if every class has zero GT) or
        ``None``; ``mAP_source`` is the provenance string surfaced in the
        summary CSV — one of ``"native evaluator"``,
        ``"unsupported backend (no Python engine to drive)"``,
        ``"failed: native evaluator crashed (see log)"``, or
        ``"no GT: every class has zero ground truth in the val set"``.
        Any crash (engine construction included) is caught and logged — the
        benchmark run continues, the row just carries no mAP.
        """
        supported = backend in ("pytorch", "onnx_fp32", "onnx_int8")
        if not supported:
            logger.warning(
                "mAP unsupported for %s: no Python engine to drive "
                "(backends without an infer() API are out of scope)", backend,
            )
            return None, "unsupported backend (no Python engine to drive)"

        from utils.map_eval import evaluate_map
        from . import YOLOv8Engine

        data_yaml = Path(self.imgs_input) / "data.yaml"
        logger.info("Computing mAP (native evaluator) for %s...", backend)

        engine = None
        try:
            engine = YOLOv8Engine(
                model_path=model_path, backend=backend,
                imgsz=self.imgsz, device=self.device,
            )

            result = evaluate_map(
                engine=engine,
                data_yaml=data_yaml,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                batch_size=self.batch_size,
                max_det=300,
                backend_name=backend,
                timestamp_suffix=self.timestamp_suffix,
                class_names_override=getattr(engine, "class_names", None),
            )
        except Exception as e:
            logger.exception("mAP computation failed for %s: %s", backend, e)
            return None, "failed: native evaluator crashed (see log)"
        finally:
            # Optional engine teardown hook (``release()``); engines without
            # one are GC-safe. ``engine`` is None when construction itself
            # raised.
            release = getattr(engine, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    logger.debug("engine.release() failed (ignored)", exc_info=True)

        if result is None:
            return None, "failed: native evaluator crashed (see log)"
        if result["mAP50"] is None:
            return result, "no GT: every class has zero ground truth in the val set"
        return result, "native evaluator"

    # pytorch path
    def _run_pytorch(self, backend: str, model_path: str) -> Dict:
        baseline = self._baseline_memory()

        # Single-stream torch thread config matching the ORT sessions.

        torch.set_num_threads(min(4, os.cpu_count() or 1))

        model = YOLO(model_path)
        model.to(self.device)
        model.model.eval()

        batches = list(self._build_batches(self.speed_imgs, self.batch_size))
        if not batches:
            raise RuntimeError("No images available for benchmarking")

        def forward(pre: Dict):
            # Raw module forward; preprocessing and NMS are handled by the
            # shared loop below.

            with torch.no_grad():
                return model.model(pre["images"])

        metrics = self._timed_end_to_end(
            backend, model_path, batches, forward, baseline
        )
        # Record the device used.
        metrics["device"] = str(self.device)
        return metrics

    # onnx path
    def _run_onnx(self, backend: str, model_path: str) -> Dict:
        baseline = self._baseline_memory()

        session = ort.InferenceSession(
            model_path,
            sess_options=_make_session_options(),
            providers=select_providers(self.device),
        )
        input_name = session.get_inputs()[0].name

        batches = list(self._build_batches(self.speed_imgs, self.batch_size))
        if not batches:
            raise RuntimeError("No images available for benchmarking")

        def forward(pre: Dict):
            inp = self._prepare_onnx_input(pre["images"])
            return session.run(None, {input_name: inp})

        metrics = self._timed_end_to_end(
            backend, model_path, batches, forward, baseline
        )
        # Record the active ORT execution provider.
        providers = session.get_providers()
        metrics["device"] = providers[0] if providers else str(self.device)
        return metrics

    def _timed_end_to_end(
        self,
        backend: str,
        model_path: str,
        batches: List[List[Path]],
        forward,
        baseline: Dict,
    ) -> Dict:
        """Warm + time the shared preprocess -> forward -> post_process loop.

        Times preprocess -> forward -> NMS per run. The NMS device follows
        each backend's output location (PT keeps predictions on CUDA; ONNX
        returns host-side numpy, so NMS runs on the host).
        """
        # --- warmup: exercise the real deployed path (incl. NMS kernels) ----
        for _ in range(self.warmup):
            pre = preprocess_imgs(
                batches[0], imgsz=self.imgsz, device=self.device
            )
            out = forward(pre)
            post_process(
                outputs=out, orig_shapes=pre["orig_shapes"],
                conf_thres=self.speed_conf, iou_thres=self.speed_iou,
                imgsz=self.imgsz, ratios=pre["ratios"], pads=pre["pads"],
            )

        if self.device == "cuda":
            # Drain warmup's async kernels before the first timed region so their
            # tail doesn't bleed into run 0; reset peak to exclude warmup.

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # --- timed runs ---
        per_run_ms: List[float] = []

        for _ in range(self.runs):
            t0 = time.perf_counter()
            for bp in batches:
                pre = preprocess_imgs(
                    bp, imgsz=self.imgsz, device=self.device
                )
                out = forward(pre)
                post_process(
                    outputs=out, orig_shapes=pre["orig_shapes"],
                    conf_thres=self.speed_conf, iou_thres=self.speed_iou,
                    imgsz=self.imgsz, ratios=pre["ratios"], pads=pre["pads"],
                )
            # CUDA kernels launch asynchronously; synchronize before reading
            # perf_counter so the delta includes GPU execution time, not just
            # CPU-side kernel-launch overhead.
            if self.device == "cuda":
                torch.cuda.synchronize()
            per_run_ms.append((time.perf_counter() - t0) * 1000.0)

        return self._aggregate_metrics(
            backend, model_path, per_run_ms, len(batches), self._peak_memory(baseline)
        )

    @staticmethod
    def _prepare_onnx_input(tensor: torch.Tensor):
        """DLPack (CUDA) when possible; otherwise numpy float32."""
        if tensor.is_cuda:
            try:
                # DLPack requires a contiguous tensor.
                return ort.OrtValue.from_dlpack(to_dlpack(tensor.contiguous()))
            except Exception:
                pass
        return tensor.detach().cpu().numpy().astype(np.float32)

    # aggregate
    def _aggregate_metrics(
        self,
        backend: str,
        model_path: str,
        per_run_ms: List[float],
        num_batches: int,
        peak_memory: Dict,
    ) -> Dict:
        n_images = len(self.speed_imgs)
        if num_batches == 0 or n_images == 0 or not per_run_ms:
            raise RuntimeError("Cannot aggregate metrics on empty benchmark set")

        run_total_s = np.asarray(per_run_ms) / 1000.0  # seconds per *run* (all batches)
        mean_total_s = float(run_total_s.mean())
        mean_batch_s = mean_total_s / num_batches
        mean_image_s = mean_total_s / n_images
        fps = n_images / mean_total_s

        pcts = percentiles(per_run_ms, qs=(50, 90, 95, 99))

        # CUDA rows: torch allocator peak. CPU rows: RSS sampled once after
        # the timed loop (semantics in the module docstring).
        peak_mb = peak_memory.get("cuda_peak_mb", peak_memory.get("system_rss_mb", 0.0))

        logger.info(
            "%s | total=%.3fs | batch=%.4fs | img=%.4fs | FPS=%.2f | "
            "p50=%.1fms p95=%.1fms p99=%.1fms | peak_mem=%.1fMB",
            backend, mean_total_s, mean_batch_s, mean_image_s, fps,
            pcts["p50"], pcts["p95"], pcts["p99"], peak_mb,
        )

        return {
            "model": os.path.basename(model_path),
            "backend": backend,
            "batch_size": self.batch_size,
            "imgsz": self.imgsz,
            "num_speed_images": n_images,
            "mean_total_s": mean_total_s,
            "mean_batch_s": mean_batch_s,
            "mean_image_s": mean_image_s,
            "fps": fps,
            "latency_ms_mean": float(np.mean(per_run_ms)),
            **pcts,
            "peak_memory_mb": float(peak_mb),
            "rss_increase_mb": float(peak_memory.get("rss_increase_mb", 0.0)),
        }

    # driver
    def run_all(self, models: Dict[str, str]) -> pd.DataFrame:
        """Run benchmarks for every backend and return a summary DataFrame."""
        results: List[Dict] = []
        summary_rows: List[List] = []

        for backend, model_path in models.items():
            logger.info("===== Benchmarking %s =====", backend)

            if backend == "pytorch":
                speed = self._run_pytorch(backend, model_path)
            else:
                speed = self._run_onnx(backend, model_path)

            # mAP provenance: per-row why mAP is present, unsupported,
            # failed, or disabled; surfaced as the `mAP_source` column in the
            # summary CSV. A backend without a Python engine reports
            # `unsupported`.
            map50 = None
            map5095 = None
            if not self.validation:
                map_source = "disabled (--validation off)"
            else:
                m, map_source = self._compute_map_native(backend, model_path)
                if m is not None and m["mAP50"] is not None:
                    map50 = m["mAP50"]
                    map5095 = m["mAP50-95"]

            results.append(speed)
            summary_rows.append([
                os.path.basename(model_path),
                backend,
                speed.get("device", ""),
                speed["batch_size"],
                speed["mean_total_s"],
                speed["mean_batch_s"],
                speed["mean_image_s"],
                speed["fps"],
                speed["latency_ms_mean"],
                speed.get("p50", 0.0),
                speed.get("p90", 0.0),
                speed.get("p95", 0.0),
                speed.get("p99", 0.0),
                speed.get("peak_memory_mb", 0.0),
                speed.get("rss_increase_mb", 0.0),
                map50,
                map5095,
                map_source,
            ])

        summary_df = pd.DataFrame(summary_rows, columns=[
            "Model", "Backend", "Device", "BatchSize", "Total_s", "Batch_s",
            "Image_s", "FPS", "Latency_ms_mean", "p50_ms", "p90_ms", "p95_ms",
            "p99_ms", "Peak_Memory_MB", "RSS_Increase_MB",
            "mAP50", "mAP50-95", "mAP_source",
        ])

        os.makedirs("results", exist_ok=True)
        summary_df.to_csv(self.summary_csv_path, index=False)
        logger.info("========== Benchmark Summary ==========")
        logger.info("\n%s", summary_df.round(3).to_string(index=False))
        return summary_df
