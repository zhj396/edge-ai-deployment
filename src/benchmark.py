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
def resolve_ort_cpp_exe() -> str:
    """Locate the built ORT-C++ backend executable.

    Search order: ``ORT_CPP_PATH`` env var (absolute path to the binary) → the
    CMake build output at ``cpp/build/onnxruntime/ort_cpp[.exe]`` relative to the
    repo root. Raises a ``RuntimeError`` pointing at the build command if
    the binary isn't found — both ``benchmark`` and ``consistency`` call this, so
    the message is the single place that tells the user to build the C++ side.
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


def _load_pybind_module(path: str, name: str):
    """Load a compiled extension by absolute path via ``importlib.util``.

    Used by :func:`resolve_trt_cpp_module` so the pybind11 backend is **never**
    a top-level import (the lazy posture means constructing the wrapper without
    the module built must not blow up — only an actual forward call resolves
    and may raise). A Python-ABI mismatch (built for Py3.11, running under
    Py3.12) surfaces as ``ImportError`` from ``exec_module`` here; we re-raise
    it as ``RuntimeError`` with a rebuild hint.
    """
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load pybind11 module from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # ABI mismatch raises ImportError here
        return mod
    except ImportError as e:
        raise RuntimeError(
            f"Failed to load {path}: {e}. This is usually a Python-ABI "
            f"mismatch — rebuild with the same interpreter you run the CLI "
            f"with (-DPython3_EXECUTABLE=$(which python))."
        ) from e


def resolve_trt_cpp_module():
    """Locate + load the built ``_trt_cpp`` pybind11 module (in-process TRT C++).

    Search order: ``TRT_CPP_PATH`` env (absolute .so/.pyd path) → the CMake
    build output at ``cpp/build/tensorrt/_trt_cpp*.so|.pyd`` (globbed — the
    SOABI-suffixed name varies by platform/Python, e.g.
    ``_trt_cpp.cpython-311-x86_64-linux-gnu.so``). Loads via ``importlib`` so
    the module is NOT a top-level import (the lazy posture means constructing
    ``TensorRTEngineCpp`` without the module built must not blow up — only an
    actual forward call resolves and may raise this). Mirrors
    :func:`resolve_ort_cpp_exe`; raises ``RuntimeError`` (never ImportError at
    module top) pointing at the build command.
    """
    env_path = os.environ.get("TRT_CPP_PATH")
    if env_path and os.path.isfile(env_path):
        return _load_pybind_module(env_path, "_trt_cpp")

    repo_root = Path(__file__).resolve().parent.parent
    build_dir = repo_root / "cpp" / "build" / "tensorrt"
    if build_dir.is_dir():
        cands = sorted(build_dir.glob("_trt_cpp*.so"))
        if platform.system() == "Windows":
            cands += sorted(build_dir.glob("_trt_cpp*.pyd"))
        # Prefer the plain-stem match (no SOABI suffix) if both exist.
        plain = [c for c in cands if c.stem == "_trt_cpp"]
        for c in (plain or cands):
            return _load_pybind_module(str(c), "_trt_cpp")

    raise RuntimeError(
        "trt_cpp pybind11 module not found. Build it first: "
        "`cmake -S cpp -B cpp/build -DBUILD_TRT_CPP=ON && "
        "cmake --build cpp/build --target _trt_cpp` (or set TRT_CPP_PATH to "
        "the .so/.pyd absolute path). Requires TensorRT + CUDA + pybind11. "
        "See docs/TENSORRT.md."
    )


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
        # (0.25 / 0.45) so the timed region measures the deployed pipeline: an
        # NMS at conf=0.001 would process all 8400 boxes and make the loop
        # NMS-bound rather than forward-bound.

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
        # that omit it get the unsuffixed fixed path (back-compat).
        self.timestamp_suffix = timestamp_suffix or ""

        # Speed-test image set: sample a diverse subset (sampler-selected) so
        # timings cover varied scenes rather than one scene type.

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

    @property
    def summary_csv_path(self) -> str:
        """Canonical summary CSV path for this run. Single-sourced so the CLI
        does not re-derive the filename pattern and drift from what
        ``run_all`` actually writes."""
        return f"results/benchmark_summary{self.timestamp_suffix}.csv"

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

    # mAP (native, backend-agnostic)
    def _compute_map_native(
        self, backend: str, model_path: str
    ) -> Tuple[Optional[Dict], str]:
        """Run the backend-agnostic COCO mAP evaluator on ``backend``.

        Constructs the engine matching ``backend`` and drives its shared
        ``infer()`` API through ``utils.map_eval.evaluate_map`` (conf=0.001,
        iou=0.7 — the COCO mAP-standard NMS config, from
        ``self.conf_threshold`` / ``self.iou_threshold``). All CLI backends go
        through this evaluator, so their mAP50 / mAP50-95 share one COCO
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
        supported = (
            backend == "pytorch"
            or backend in ("onnx_fp32", "onnx_int8")
            or backend.startswith("openvino")
            or backend.startswith("tensorrt")
            or backend.startswith("trt_cpp")
        )
        if not supported:
            logger.warning(
                "mAP unsupported for %s: no Python engine to drive "
                "(backends without an infer() API are out of scope)", backend,
            )
            return None, "unsupported backend (no Python engine to drive)"

        from utils.map_eval import evaluate_map
        from . import (
            YOLOv8Engine,
            OpenVINOEngine,
            openvino_available,
            TensorRTEngine,
            tensorrt_available,
            TensorRTEngineCpp,
            trt_cpp_available,
        )

        data_yaml = Path(self.imgs_input) / "data.yaml"
        logger.info("Computing mAP (native evaluator) for %s...", backend)

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
            elif backend.startswith("tensorrt"):
                if not tensorrt_available():
                    raise RuntimeError(
                        "tensorrt is not installed. Run: "
                        "pip install -r requirements-tensorrt.txt"
                    )
                # The TRT engine always runs on GPU (--device is irrelevant
                # here; it selects the GPU id only in the tensorrt build/run
                # CLI). max_batch mirrors _run_tensorrt.
                engine = TensorRTEngine(
                    model_path=model_path, imgsz=self.imgsz,
                    max_batch=max(self.batch_size, 8), data_yaml=data_yaml,
                )
            elif backend.startswith("trt_cpp"):
                # In-process C++ pybind11 backend: SAME .engine + SAME metric
                # surface as the Python tensorrt* rows, so the trt_cpp vs tensorrt
                # mAP delta isolates the binding overhead. Native mAP, NOT skipped
                # (unlike ort_cpp, which is a subprocess exe with no Python engine).
                if not trt_cpp_available():
                    raise RuntimeError(
                        "trt_cpp pybind11 module not built. Build it first: "
                        "`cmake -S cpp -B cpp/build -DBUILD_TRT_CPP=ON && "
                        "cmake --build cpp/build --target _trt_cpp` (or set "
                        "TRT_CPP_PATH). See docs/TENSORRT.md."
                    )
                engine = TensorRTEngineCpp(
                    model_path=model_path, imgsz=self.imgsz,
                    max_batch=max(self.batch_size, 8), data_yaml=data_yaml,
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
            # TRT owns CUDA buffers + a retained primary context that need
            # explicit teardown (mirrors _run_tensorrt's finally). OV/ONNX/PT
            # engines are GC-safe — release() is a no-op / absent.
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

    # ort_cpp path
    def _run_ort_cpp(self, backend: str, model_path: str) -> Dict:
        """ORT-C++ backend via the built ``ort_cpp`` exe (subprocess).

        The C++ benchmark times ``detect()`` = preprocess + infer + postprocess + NMS — the same
        scope as the Python ``_timed_end_to_end`` loop, so the headline FPS/latency are
        comparable with the Python ORT row. It is **single-image** (batch=1): the C++ exe
        has no batched-forward path, so ``batch_size`` is reported as 1 regardless of
        ``--batch-size`` (the requested value is kept alongside). The image set is
        copied into a temp dir with zero-padded names so the C++ ``--dir`` loader iterates the
        *same* image set as the Python backends, in the same order.

        The C++ exe reports **per-image** mean/percentiles/throughput. The Python rows report
        **per-sweep** (one sweep = all batches) latency/percentiles. To keep the summary CSV
        comparable across rows, the C++ per-image numbers are scaled to sweep units
        (× ``n_images``) for ``mean_total_s`` / ``latency_ms_mean`` / ``p50-p99``; ``fps`` and
        ``mean_image_s`` are per-image and directly comparable. The sweep-scaled percentiles are
        an approximation (assumes similar per-image latency across the set).
        """
        exe = resolve_ort_cpp_exe()
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
                # Match _make_session_options (intra=4, inter=2) so the C++-vs-Python-ORT
                # comparison is on the same single-stream thread footing.
                "--intra-op-threads", str(min(4, os.cpu_count() or 1)),
                "--inter-op-threads", "2",
                "--save-json", tmp_json,
            ]
            logger.info("ort_cpp cmd: %s", " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True)
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
                "latency_ms_mean": mean_ms * n_images,   # sweep latency (matches Python rows)
                "p50": sweep("p50_ms"),
                "p90": 0.0,
                "p95": sweep("p95_ms"),
                "p99": sweep("p99_ms"),
                # The C++ process reports absolute peak RSS, not a delta from a pre-load
                # baseline (it is a separate process). rss_increase_mb=0 — documented.
                "peak_memory_mb": float(j.get("peak_rss_mb", 0.0)),
                "rss_increase_mb": 0.0,
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

    # tensorrt path
    def _run_tensorrt(self, backend: str, model_path: str) -> Dict:
        """TensorRT serialized engine via the Python backend (cuda-python)."""
        from . import TensorRTEngine, tensorrt_available
        return self._run_tensorrt_family(
            backend, model_path, TensorRTEngine, tensorrt_available,
            "tensorrt is not installed; run: "
            "pip install -r requirements-tensorrt.txt",
        )

    def _run_trt_cpp(self, backend: str, model_path: str) -> Dict:
        """TensorRT serialized engine via the in-process C++ pybind11 backend.

        Same ``.engine`` and same metric surface as :meth:`_run_tensorrt`
        (native mAP via ``_compute_map_native`` + ``kernel_latency_ms``), so the
        ``trt_cpp*`` vs ``tensorrt*`` comparison isolates the C++-binding vs
        cuda-python overhead — the reason this backend exists.
        """
        from . import TensorRTEngineCpp, trt_cpp_available
        return self._run_tensorrt_family(
            backend, model_path, TensorRTEngineCpp, trt_cpp_available,
            "trt_cpp pybind11 module not built. Build it first: "
            "`cmake -S cpp -B cpp/build -DBUILD_TRT_CPP=ON && "
            "cmake --build cpp/build --target _trt_cpp` (or set TRT_CPP_PATH). "
            "See docs/TENSORRT.md.",
        )

    def _run_tensorrt_family(
        self, backend: str, model_path: str,
        ctor, avail_fn, unavail_msg: str,
    ) -> Dict:
        """Shared timed-loop body for the TensorRT-family backends (``tensorrt*``
        Python + ``trt_cpp*`` in-process C++). Both engines mirror the same API
        (``_effective_batch`` / ``_forward`` / ``kernel_timed_forward`` /
        ``release``), so the body is single-sourced; the caller passes the
        constructor + availability predicate. Mirrors :meth:`_run_openvino`:
        construct once, feed the shared preprocess -> forward -> post_process
        pipeline, then ``release()`` in ``finally``.

        Two extra metrics are attached for these rows only:
        ``kernel_latency_ms`` / ``kernel_fps`` — the GPU forward pass
        (H2D+execute+D2H) free of Python NMS / letterbox / disk I/O, via
        ``kernel_timed_forward``. This preserves the TRT project's standout
        metric without breaking cross-backend end-to-end parity (other rows
        omit the two columns).
        """
        from . import TensorRTEngine, tensorrt_available

        baseline = self._baseline_memory()

        if not tensorrt_available():
            raise RuntimeError(
                "tensorrt is not installed; run: "
                "pip install -r requirements-tensorrt.txt"
            )
        engine = TensorRTEngine(
            model_path=model_path, imgsz=self.imgsz,
            max_batch=max(self.batch_size, 8),
        )
        # Cap to the engine's profile max_batch: a forward exceeding it is
        # rejected by TRT. The headline FPS/latency stay correct either way.
        eff = engine._effective_batch(self.batch_size)

        batches = list(self._build_batches(self.speed_imgs, eff))
        if not batches:
            raise RuntimeError("No images available for benchmarking")

        def forward(pre: Dict):
            return engine._forward(pre["images"].cpu().numpy())

        try:
            metrics = self._timed_end_to_end(
                backend, model_path, batches, forward, baseline
            )

            # GPU-kernel-only timing — a short loop over the first batch,
            # timed around just the H2D+execute+D2H window (no NMS / preprocess
            # in the timed region). The end-to-end numbers above already
            # include NMS.
            pre0 = preprocess_imgs(batches[0], imgsz=self.imgsz, device="cpu")
            _ = engine.kernel_timed_forward(pre0["images"].cpu().numpy())  # warm
            k_ms = []
            for _ in range(min(self.runs, 25)):
                _raw, ms = engine.kernel_timed_forward(
                    pre0["images"].cpu().numpy()
                )
                k_ms.append(ms)
            metrics["kernel_latency_ms"] = float(np.mean(k_ms))
            metrics["kernel_fps"] = (
                1000.0 / metrics["kernel_latency_ms"]
                if metrics["kernel_latency_ms"] > 0 else 0.0
            )
            logger.info(
                "%s | kernel=%.2fms | kernel_fps=%.1f (GPU forward only, no NMS)",
                backend, metrics["kernel_latency_ms"], metrics["kernel_fps"],
            )
        finally:
            engine.release()

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
            elif backend == "ort_cpp":
                speed = self._run_ort_cpp(backend, model_path)
            elif backend.startswith("openvino"):
                speed = self._run_openvino(backend, model_path)
            elif backend.startswith("tensorrt"):
                speed = self._run_tensorrt(backend, model_path)
            elif backend.startswith("trt_cpp"):
                speed = self._run_trt_cpp(backend, model_path)
            else:
                speed = self._run_onnx(backend, model_path)

            # mAP provenance: track per-row why mAP is present, unsupported,
            # failed, or disabled; surfaced as the `mAP_source` column in the
            # summary CSV. All CLI backends run through one native COCO
            # evaluator (utils.map_eval over each engine's infer()) so their
            # mAP50 / mAP50-95 share one metric and are comparable across
            # backends. A backend without a Python engine to drive reports
            # `unsupported`; its precision signal is the cross-backend
            # consistency harness (src/consistency.py — raw-tensor allclose
            # + conf-cliff gate).
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
            "Image_s", "FPS", "Latency_ms_mean", "p50_ms", "p95_ms",
            "p99_ms", "Peak_Memory_MB", "RSS_Increase_MB",
            "mAP50", "mAP50-95", "mAP_source",
        ])

        os.makedirs("results", exist_ok=True)
        summary_df.to_csv(self.summary_csv_path, index=False)
        logger.info("========== Benchmark Summary ==========")
        logger.info("\n%s", summary_df.round(3).to_string(index=False))
        return summary_df
