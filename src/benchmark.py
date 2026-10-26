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
import json
import os
import platform
import shutil
import subprocess
import tempfile
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
# Suffixes the C++ --dir loader decodes (cpp/onnxruntime/src/main.cpp
# ::is_image_extension); anything else is silently dropped there. Both ort_cpp
# staging loops (benchmark + consistency) reject other suffixes up front.
ORT_CPP_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def resolve_ort_cpp_exe() -> str:
    """Locate the built ORT-C++ backend executable.

    Search order: ``ORT_CPP_PATH`` env var (absolute path to the binary) → the
    CMake build output at ``cpp/build/onnxruntime/ort_cpp[.exe]`` relative to the
    repo root. Raises a ``RuntimeError`` with the build command if the binary
    is not found.
    """
    env_path = os.environ.get("ORT_CPP_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    exe_name = "ort_cpp.exe" if platform.system() == "Windows" else "ort_cpp"
    # Repo root = this file's parent's parent (src/ -> edge-ai-deployment/).
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "cpp" / "build" / "onnxruntime" / exe_name
    if candidate.is_file():
        return str(candidate)

    raise RuntimeError(
        "ORT-C++ backend executable not found. Build it first: "
        "`cmake --build cpp/build` (or set ORT_CPP_PATH to the binary's "
        "absolute path). See docs/ORT_CPP.md."
    )


def _thread_knobs() -> tuple:
    """The (intra, inter) thread values ``_make_session_options`` bakes in."""
    ncpu = os.cpu_count() or 1
    return min(4, ncpu), (2 if min(4, ncpu) > 1 else 1)


def _make_session_options() -> ort.SessionOptions:
    """ORT session options: single-stream thread config (intra=min(4, ncpu);
    inter=2 when intra>1, else 1)."""
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
        with a Python engine goes through this evaluator, so their mAP50 /
        mAP50-95 share one COCO metric and are directly comparable across
        those backends.

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
        supported = (
            backend == "pytorch"
            or backend in ("onnx_fp32", "onnx_int8")
            or backend.startswith("openvino")
        )
        if not supported:
            logger.warning(
                "mAP unsupported for %s: no Python engine to drive "
                "(backends without an infer() API are out of scope)", backend,
            )
            return None, "unsupported backend (no Python engine to drive)"

        from utils.map_eval import evaluate_map
        from . import YOLOv8Engine, OpenVINOEngine, openvino_available

        data_yaml = Path(self.imgs_input) / "data.yaml"
        logger.info("Computing mAP (native evaluator) for %s...", backend)

        engine = None
        try:
            if backend == "pytorch":
                engine = YOLOv8Engine(
                    model_path=model_path, backend="pytorch",
                    imgsz=self.imgsz, device=self.device,
                )
            elif backend in ("onnx_fp32", "onnx_int8"):
                engine = YOLOv8Engine(
                    model_path=model_path, backend=backend,
                    imgsz=self.imgsz, device=self.device,
                )
            elif backend.startswith("openvino"):
                if not openvino_available():
                    raise RuntimeError(
                        "openvino is not installed. Run: "
                        "pip install -r requirements-openvino.txt"
                    )
                # Device follows OPENVINO_DEVICE (default CPU) — same posture
                # as _run_openvino (--device is irrelevant for OpenVINO since
                # it takes host numpy).
                ov_device = os.environ.get("OPENVINO_DEVICE", "CPU")
                engine = OpenVINOEngine(
                    model_path=model_path, device=ov_device,
                    imgsz=self.imgsz, data_yaml=data_yaml,
                )
            # ort_cpp is a subprocess exe with no Python engine, so its row
            # skips the native evaluator.

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

        # A fixed-shape session only accepts its static batch dim: run that
        # size and log when it differs from the requested --batch-size (same
        # behavior as YOLOv8Engine._effective_batch).
        in_shape = session.get_inputs()[0].shape
        batch_size = self.batch_size
        if isinstance(in_shape[0], int) and in_shape[0] > 0 \
                and in_shape[0] != self.batch_size:
            batch_size = in_shape[0]
            logger.warning(
                "Model input has static batch %d; using batch size %d "
                "(requested %d)",
                in_shape[0], batch_size, self.batch_size,
            )

        batches = list(self._build_batches(self.speed_imgs, batch_size))
        if not batches:
            raise RuntimeError("No images available for benchmarking")

        def forward(pre: Dict):
            inp = self._prepare_onnx_input(pre["images"])
            return session.run(None, {input_name: inp})

        metrics = self._timed_end_to_end(
            backend, model_path, batches, forward, baseline
        )
        # The CSV row shows the batch size actually run (a static batch dim
        # overrides the requested --batch-size above).
        metrics["batch_size"] = batch_size
        # Record the active ORT execution provider.
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
        # reshape. The reported FPS/latency stay correct either way.
        eff = engine._effective_batch(self.batch_size)

        batches = list(self._build_batches(self.speed_imgs, eff))
        if not batches:
            raise RuntimeError("No images available for benchmarking")

        def forward(pre: Dict):
            return engine._forward(pre["images"].cpu().numpy())

        metrics = self._timed_end_to_end(
            backend, model_path, batches, forward, baseline
        )
        # Record the device that *actually* executed, not the
        # OPENVINO_DEVICE request: engine.device is the resolved value from
        # validate_device_request's preflight, so an unservable request
        # (e.g. GPU without an Intel GPU driver) that fell back to CPU is
        # reported as CPU. Keep the request alongside so the fallback stays
        # visible in the results.
        metrics["device"] = engine.device
        if engine.device != device.upper():
            metrics["requested_device"] = device.upper()
        # When a static-batch IR capped eff below the requested
        # --batch-size, "batch_size" reports what *actually* ran per forward
        # (eff); keep the requested value alongside.
        if eff != self.batch_size:
            metrics["requested_batch_size"] = self.batch_size
            metrics["batch_size"] = eff
        return metrics

    # ort_cpp path
    def _run_ort_cpp(self, backend: str, model_path: str) -> Dict:
        """ORT-C++ backend via the built ``ort_cpp`` exe (subprocess).

        The C++ benchmark times ``detect()`` = preprocess + infer + postprocess + NMS on
        already-decoded images. The Python ``_timed_end_to_end`` loop times preprocess
        (including image decode) → forward → NMS, so the Python rows carry decode cost the
        C++ rows do not; the C++ protocol is also per-iteration while the Python loop is
        per-sweep, so treat cross-protocol rows as
        indicative. The C++ exe is **single-image** (batch=1): it
        has no batched-forward path, so ``batch_size`` is reported as 1 regardless of
        ``--batch-size`` (the requested value is kept as the ``requested_batch_size``
        metrics key; it is not a summary-CSV column). The image set is
        copied into a temp dir with zero-padded names so the C++ ``--dir`` loader iterates the
        *same* image set as the Python backends, in the same order.

        The C++ exe reports **per-image** mean/percentiles/throughput. The Python rows report
        **per-sweep** (one sweep = all batches) latency/percentiles. To keep the summary CSV
        comparable across rows, the C++ per-image numbers are scaled to sweep units
        (× ``n_images``) for ``mean_total_s`` / ``latency_ms_mean`` / ``p50`` / ``p95`` /
        ``p99`` (the C++ exe reports no p90; that metric stays 0); ``fps`` and
        ``mean_image_s`` are per-image and directly comparable. The sweep-scaled percentiles are
        an approximation (assumes similar per-image latency across the set).
        """
        exe = resolve_ort_cpp_exe()
        intra, inter = _thread_knobs()
        n_images = len(self.speed_imgs)
        if n_images == 0:
            raise RuntimeError("No images available for benchmarking")

        self._baseline_memory()  # warm the baseline path for parity (RSS comes from the C++ proc)

        tmp_in = tempfile.mkdtemp(prefix="ort_cpp_bench_in_")
        out_dir = tempfile.mkdtemp(prefix="ort_cpp_bench_out_")
        tmp_json = os.path.join(out_dir, "summary.json")
        try:
            width = max(6, len(str(n_images - 1)))
            for i, p in enumerate(self.speed_imgs):
                ext = Path(str(p)).suffix or ".jpg"
                # The C++ --dir loader silently drops suffixes outside
                # ORT_CPP_IMAGE_SUFFIXES; reject them at staging time so the
                # sweep-scaled metrics below cover all n_images.
                if ext.lower() not in ORT_CPP_IMAGE_SUFFIXES:
                    raise RuntimeError(
                        f"ort_cpp backend requires .jpg/.jpeg/.png/.bmp speed "
                        f"images, got '{ext}': {p}"
                    )
                shutil.copyfile(str(p), os.path.join(tmp_in, f"{i:0{width}d}{ext}"))

            cmd = [
                exe,
                "--benchmark", str(self.runs),
                "--dir", tmp_in,
                "--max-images", str(n_images),
                "-m", model_path,
                "--imgsz", str(self.imgsz),
                "--conf", str(self.speed_conf),
                "--iou", str(self.speed_iou),
                # Thread knobs matching the Python ORT sessions. Warmup is not
                # forwarded: the driver warms up a fixed 3 iterations before
                # its timed loop (see docs/ORT_CPP.md).
                "--intra-op-threads", str(intra),
                "--inter-op-threads", str(inter),
                "--save-json", tmp_json,
            ]
            logger.info("ort_cpp cmd: %s", " ".join(cmd))
            # Fixed 600 s timeout: a hung exe (stale build, deadlocked
            # session) fails the run at the cap.
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ort_cpp benchmark failed (exit {proc.returncode}):\n"
                    f"stderr: {proc.stderr[-2000:]}"
                )
            with open(tmp_json, "r", encoding="utf-8") as f:
                j = json.load(f)

            mean_ms = float(j.get("mean_ms", 0.0))
            per_image_s = mean_ms / 1000.0
            sweep_s = per_image_s * n_images  # one sweep = n_images forwards (batch=1)
            fps = (1000.0 / mean_ms) if mean_ms > 0 else 0.0
            # Scale per-image percentiles to sweep units so the CSV columns match the Python rows.
            sweep = lambda k: float(j.get(k, 0.0)) * n_images  # noqa: E731

            metrics = {
                "model": os.path.basename(model_path),
                "backend": backend,
                "batch_size": 1,
                "requested_batch_size": self.batch_size,
                "imgsz": self.imgsz,
                "num_speed_images": n_images,
                "mean_total_s": sweep_s,
                "mean_batch_s": per_image_s,   # batch=1 → per-batch == per-image
                "mean_image_s": per_image_s,
                "fps": fps,
                # sweep-scaled (per-image × n_images); Python rows are
                # per-sweep wallclock.
                "latency_ms_mean": mean_ms * n_images,
                "p50": sweep("p50_ms"),
                "p90": 0.0,
                "p95": sweep("p95_ms"),
                "p99": sweep("p99_ms"),
                # The C++ process reports absolute peak RSS, not a delta from a pre-load
                # baseline (it is a separate process). rss_increase_mb=0 — documented.
                "peak_memory_mb": float(j.get("peak_rss_mb", 0.0)),
                "rss_increase_mb": 0.0,
                # The exe's own platform: this backend runs in a separate
                # process, so the CSV Device column carries the exe host;
                # --device does not apply to this backend.
                "device": f"ort_cpp ({platform.system()})",
            }
            logger.info(
                "%s | img=%.4fs | FPS=%.2f | p50=%.1fms p95=%.1fms p99=%.1fms | "
                "peak_mem=%.1fMB (per-image → sweep-scaled)",
                backend, per_image_s, fps,
                sweep("p50_ms"), sweep("p95_ms"), sweep("p99_ms"),
                metrics["peak_memory_mb"],
            )
            return metrics
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

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
                ort_input = ort.OrtValue.from_dlpack(to_dlpack(tensor.contiguous()))
                # Same barrier as YOLOv8Engine._forward and
                # consistency.ort_forward: the tensor is produced
                # asynchronously on the torch stream while ORT executes on its
                # own stream and reads the DLPack buffer without waiting for
                # the producer. Synchronize the producing stream before
                # handing the buffer to ORT.
                torch.cuda.current_stream().synchronize()
                return ort_input
            except Exception as e:
                # The fallback adds a host copy to the timed path — log it so
                # the measurement mode is visible in the run's provenance.
                logger.warning(
                    "DLPack OrtValue input failed (%s: %s); falling back to "
                    "host numpy for the benchmark loop.",
                    type(e).__name__, e,
                )
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
        """Run benchmarks for every backend and return a summary DataFrame.

        A backend whose run raises (an engine crash, a missing optional
        runtime) is logged and omitted from the results — the loop
        continues, and the closing warning names every failed backend with
        its reason (the same omit-and-summarize contract as
        ``OpenVINOEngine.infer``'s failed batches). The summary CSV covers
        the backends that ran.
        """
        results: List[Dict] = []
        summary_rows: List[List] = []
        failed_backends: List[tuple] = []

        for backend, model_path in models.items():
            logger.info("===== Benchmarking %s =====", backend)

            try:
                if backend == "pytorch":
                    speed = self._run_pytorch(backend, model_path)
                elif backend == "ort_cpp":
                    speed = self._run_ort_cpp(backend, model_path)
                elif backend.startswith("openvino"):
                    speed = self._run_openvino(backend, model_path)
                else:
                    speed = self._run_onnx(backend, model_path)

                # mAP provenance: per-row why mAP is present, unsupported,
                # failed, or disabled; surfaced as the `mAP_source` column in the
                # summary CSV. A backend without a Python engine reports
                # `unsupported`; its precision signal is the consistency harness.
                map50 = None
                map5095 = None
                if not self.validation:
                    map_source = "disabled (--validation off)"
                else:
                    m, map_source = self._compute_map_native(backend, model_path)
                    if m is not None and m["mAP50"] is not None:
                        map50 = m["mAP50"]
                        map5095 = m["mAP50-95"]
            except Exception as e:
                # Per-backend isolation: one backend's crash (engine
                # construction, missing optional runtime, timed loop) must
                # not discard the measurements of the other backends.
                failed_backends.append((backend, model_path, e))
                logger.exception(
                    "Benchmark failed for %s (model=%s)", backend, model_path,
                )
                continue

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

        if failed_backends:
            logger.warning(
                "%d/%d backends failed and were skipped from the summary: %s",
                len(failed_backends), len(models),
                [(b, str(e)) for b, _, e in failed_backends],
            )

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
