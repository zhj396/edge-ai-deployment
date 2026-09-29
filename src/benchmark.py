"""Performance benchmark for YOLOv8s backends.

The benchmark measures end-to-end inference latency, throughput (FPS), and peak memory across
PyTorch and ONNX Runtime backends.

Measurement notes
-----------------
* **Percentile latencies** (p50/p90/p95/p99) are reported alongside the mean — tail latency
matters in production, not the mean.
* Memory tracking is backend-aware: CUDA peak memory via ``torch.cuda.max_memory_allocated``; CPU
RSS via ``psutil``.
* Each backend run starts from a known memory baseline so the RSS-increase number reflects only that
backend's footprint.
* ``malloc_trim`` is invoked on Linux between runs to return freed pages to the OS — otherwise the
RSS baseline keeps climbing.
"""
from __future__ import annotations

import ctypes
import gc
import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

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
def _make_session_options() -> ort.SessionOptions:
    """Single-stream ORT thread tuning: intra=4, inter=2.

    Without this ORT defaults to "all cores", which oversubscribes a single-stream YOLOv8s CPU
    inference and inflates latency. Matches the engine's defaults so benchmark and infer are
    measured on the same footing.
    """
    ncpu = os.cpu_count() or 1
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = min(4, ncpu)
    opts.inter_op_num_threads = 2 if min(4, ncpu) > 1 else 1
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
    ) -> None:
        self.imgs_input = imgs_input
        self.max_images_for_speed = max_images_for_speed
        self.imgsz = imgsz
        self.batch_size = batch_size
        # conf/iou_threshold drive the optional mAP *validation* (model.val), which needs
        # COCO-standard conf->0 / iou=0.7 for a correct PR curve. speed_conf/speed_iou drive
        # the NMS inside the *speed* loop and default to engine.infer's deploy values (0.25 / 0.45)
        # so the timed region measures the real deployed pipeline, not an NMS-at-conf=0.001 path
        # that would process all 8400 boxes and be NMS-bound rather than forward-bound.

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

        # Speed-test image set: use a diverse subset rather than the first N images (avoids
        # over-warm caches and bias toward similar scenes).

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
                    "paths instead (less representative, ~free cost).", e,
                )
                self.speed_imgs = self._load_val_paths(
                    data_yaml, max_images_for_speed
                )
        else:
            # Explicit opt-out: skip ResNet-50 entirely. Use when you want a quick
            # speed check and don't care about class-stratified sampling.

            self.speed_imgs = self._load_val_paths(
                data_yaml, max_images_for_speed
            )
        logger.info("Speed test set: %d images", len(self.speed_imgs))

    @staticmethod
    def _load_val_paths(data_yaml: Path, n: int) -> List[Path]:
        """Cheap path-only speed-test set: read ``data.yaml``'s ``val`` dir and
        return the first ``n`` sorted image paths.

        No label parsing, no ResNet-50 — used when CalibrationSampler is disabled
        (``use_sampler=False``) or when it raises. Representativeness is weaker than the sampler's
        stratified + farthest-first selection (just sorted filenames), but the cost is ~free and it
        has no heavy dependencies, so the benchmark runs on any box with the dataset.
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

    # mAP
    def _compute_map(self, model: YOLO, backend_name: str):
        """Run Ultralytics ``model.val`` to get per-class mAP."""
        logger.info("Computing mAP for %s on full validation set...", backend_name)
        data_yaml = Path(self.imgs_input) / "data.yaml"

        try:
            results = model.val(
                data=str(data_yaml),
                imgsz=self.imgsz,
                batch=self.batch_size,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                save=False,
                plots=False,
                save_json=False,
                verbose=False,
            )
        except Exception as e:
            logger.exception("mAP computation failed for %s: %s", backend_name, e)
            return None

        rows = []
        for cls_metrics in results.summary():
            cls_name = cls_metrics.get("Class") or cls_metrics.get("class") or "unknown"
            rows.append([
                backend_name,
                cls_name,
                float(cls_metrics.get("mAP50", cls_metrics.get("map50", 0.0))),
                float(cls_metrics.get("mAP50-95", cls_metrics.get("map", 0.0))),
            ])
            logger.info(
                "%s | %s: mAP50=%.4f, mAP50-95=%.4f",
                backend_name, cls_name, rows[-1][2], rows[-1][3],
            )

        try:
            p, r, map50, map5095 = results.mean_results()
            logger.info(
                "%s Overall | P=%.4f, R=%.4f, mAP50=%.4f, mAP50-95=%.4f",
                backend_name, p, r, map50, map5095,
            )
        except Exception as e:
            logger.warning("%s overall metrics parse failed: %s", backend_name, e)

        os.makedirs("results", exist_ok=True)
        csv_path = f"results/{backend_name}_perclass.csv"
        pd.DataFrame(rows, columns=["backend", "class", "mAP50", "mAP50-95"]) \
            .to_csv(csv_path, index=False)
        logger.info("%s per-class metrics saved: %s", backend_name, csv_path)
        return results

    # pytorch path
    def _run_pytorch(self, backend: str, model_path: str) -> Dict:
        baseline = self._baseline_memory()

        # Match the ORT path's single-stream thread tuning so the PyTorch-vs-ORT
        # comparison is apples-to-apples (otherwise torch uses all cores).

        torch.set_num_threads(min(4, os.cpu_count() or 1))

        model = YOLO(model_path)
        model.to(self.device)
        model.model.eval()

        batches = list(self._build_batches(self.speed_imgs, self.batch_size))
        if not batches:
            raise RuntimeError("No images available for benchmarking")

        def forward(pre: Dict):
            # Raw module forward — NOT Ultralytics' ``model(bp)`` predict path (which
            # does its own letterbox + NMS + Results formatting). Both backends share
            # preprocess_imgs + post_process so the timed region measures identical
            # end-to-end work, matching ``YOLOv8Engine.infer`` (preprocess -> forward
            # -> NMS). No autocast: the ONNX export is FP32, so the PT path stays FP32
            # for a fair PT-vs-ONNX comparison.

            with torch.no_grad():
                return model.model(pre["images"])

        metrics = self._timed_end_to_end(
            backend, model_path, batches, forward, baseline
        )
        # Reproducibility (CLAUDE.md): record the device that actually ran.
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
        # Reproducibility (CLAUDE.md): record the *actual* primary execution
        # provider — ORT itself may have fallen back from the requested EP
        # (e.g. CUDA unavailable -> CPU), so get_providers() is more honest
        # than echoing self.device.
        providers = session.get_providers()
        metrics["device"] = providers[0] if providers else str(self.device)
        return metrics

    # openvino path
    def _run_openvino(self, backend: str, model_path: str) -> Dict:
        """OpenVINO IR / ONNX via the OpenVINO runtime.

        The OpenVINO *device* (CPU / GPU / AUTO) is selected via the
        ``OPENVINO_DEVICE`` env var, not ``--device`` — ``--device`` still
        controls only where preprocess runs (and is irrelevant here: OpenVINO
        takes host numpy, so we always preprocess on CPU). Mirrors the
        ``_run_onnx`` shape: compile once, then the timed loop feeds the
        shared preprocess -> forward -> post_process pipeline.
        """
        baseline = self._baseline_memory()

        from . import OpenVINOEngine, openvino_available

        if not openvino_available():
            raise RuntimeError(
                "OpenVINO not installed; run: pip install -r requirements-openvino.txt"
            )
        device = os.environ.get("OPENVINO_DEVICE", "CPU")
        engine = OpenVINOEngine(
            model_path=model_path, device=device, imgsz=self.imgsz,
        )
        # Cap to the model's per-forward batch: a static-batch IR (e.g. one
        # converted from a static-batch ONNX) rejects batch>1 at the DFL
        # reshape. The headline FPS/latency stay correct either way.
        eff = engine._effective_batch(self.batch_size)

        batches = list(self._build_batches(self.speed_imgs, eff))
        if not batches:
            raise RuntimeError("No images available for benchmarking")

        def forward(pre: Dict):
            return engine._forward(pre["images"].cpu().numpy())

        metrics = self._timed_end_to_end(
            backend, model_path, batches, forward, baseline
        )
        # Reproducibility (CLAUDE.md): record the device that *actually*
        # executed, not the OPENVINO_DEVICE request. engine.device is the
        # resolved value from validate_device_request's preflight — when an
        # unservable request (e.g. GPU without an Intel GPU driver) fell
        # back to CPU, the metrics must say CPU or the summary would
        # advertise iGPU throughput measured on the host CPU. Keep the
        # request alongside so the fallback stays visible in the results.
        metrics["device"] = engine.device
        if engine.device != device.upper():
            metrics["requested_device"] = device.upper()
        # When a static-batch IR capped eff below the requested --batch-size,
        # "batch_size" must report what *actually* ran per forward (eff), not
        # the requested value — otherwise the summary CSV advertises batched
        # throughput that never happened (the static IR was sub-looped at
        # its baked batch). Keep the requested value alongside for honesty.
        if eff != self.batch_size:
            metrics["requested_batch_size"] = self.batch_size
            metrics["batch_size"] = eff
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

        Both backends run the *same* stages here (letterbox preprocess, raw forward, NMS via
        Ultralytics + scale-back), matching ``YOLOv8Engine.infer`` — so the headline PT-vs-ONNX
        FPS comparison measures identical end-to-end work on both sides, NMS included.

        Note: the device the NMS runs on differs by backend, and this is intentional — it reflects
        real deployment. The PT path keeps the prediction on CUDA and NMS runs there; the ONNX CUDA
        path returns host-side numpy from ``session.run`` (or ``copy_outputs_to_cpu`` in the
        engine), so NMS runs on the host. That asymmetry is exactly what ``engine.infer`` does too.
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
            # CUDA kernels launch asynchronously; without a synchronize the perf_counter
            # delta only captures CPU-side launch overhead, not real GPU execution time.
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
                # ``.contiguous()`` is a no-op for the already-contiguous preprocess
                # output and prevents an opaque DLPack error on any future
                # non-contiguous input (slice/permute).
                # Matches ``YOLOv8Engine._forward``'s CUDA branch.
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
        if num_batches == 0 or n_images == 0:
            raise RuntimeError("Cannot aggregate metrics on empty benchmark set")

        run_total_s = np.asarray(per_run_ms) / 1000.0  # seconds per *run* (all batches)
        mean_total_s = float(run_total_s.mean())
        mean_batch_s = mean_total_s / num_batches
        mean_image_s = mean_total_s / n_images
        fps = n_images / mean_total_s

        pcts = percentiles(per_run_ms, qs=(50, 90, 95, 99))

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
            elif backend.startswith("openvino"):
                speed = self._run_openvino(backend, model_path)
            else:
                speed = self._run_onnx(backend, model_path)

            if self.validation:
                if backend.startswith("openvino"):
                    # Ultralytics' YOLO() can load its own OpenVINO export
                    # (an openvino_model/ folder) but not a raw .xml produced
                    # by our converter — it lacks the ultralytics metadata.
                    # Skip rather than crash; run validation on the .pt if mAP
                    # is needed.
                    logger.warning(
                        "mAP validation skipped for %s (raw OpenVINO IR not "
                        "loadable by Ultralytics YOLO; validate the .pt instead)",
                        backend,
                    )
                else:
                    val_model = YOLO(model_path, task='detect')
                    self._compute_map(val_model, backend)

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
                speed.get("p95", 0.0),
                speed.get("p99", 0.0),
                speed.get("peak_memory_mb", 0.0),
                speed.get("rss_increase_mb", 0.0),
            ])

        summary_df = pd.DataFrame(summary_rows, columns=[
            "Model", "Backend", "Device", "BatchSize", "Total_s", "Batch_s",
            "Image_s", "FPS", "Latency_ms_mean", "p50_ms", "p95_ms",
            "p99_ms", "Peak_Memory_MB", "RSS_Increase_MB",
        ])

        os.makedirs("results", exist_ok=True)
        summary_df.to_csv("results/benchmark_summary.csv", index=False)
        logger.info("========== Benchmark Summary ==========")
        logger.info("\n%s", summary_df.round(3).to_string(index=False))
        return summary_df
