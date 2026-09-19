# YOLOv8s Multi-Backend Inference Toolkit — ONNX Runtime · INT8 PTQ · Benchmark · Consistency Validation

[![CI](https://github.com/zhj396/edge-ai-deployment/actions/workflows/ci.yml/badge.svg)](https://github.com/zhj396/edge-ai-deployment/actions/workflows/ci.yml)

A production-oriented Python toolchain for taking a **YOLOv8s** PyTorch checkpoint through the full edge-deployment lifecycle: **export → ONNX Runtime inference → static INT8 quantization (QDQ / MinMax or Entropy) → consistency & accuracy validation → latency / throughput / memory / mAP benchmark**, all reproducible from a single CLI.

Built and validated end-to-end on a **local PC (13th Gen Intel Core i5-13420H)** and a **Kaggle Tesla T4 ×2 GPU node**, with no embedded target board — sampling strategies and ORT knobs are fully documented; benchmark figures are tracked in [docs/ARCHITECTURE.md §11](docs/ARCHITECTURE.md#11-known-results-representative-numbers).

> A full architecture deep-dive — pipeline, quantization policy, consistency gates, benchmarking — lives in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.
>
> The upstream training half — COCO 12-class subset build, dataset analysis, and baseline vs. staged long-tail training that produces `models/yolov8s.pt` — lives in [`train/`](train/) with the full narrative in **[docs/TRAINING.md](docs/TRAINING.md)**.
>
> Docker packaging — a `toolchain` image (the full CLI) and a `server` image (FastAPI inference, CPU) — is covered in **[docs/DOCKER.md](docs/DOCKER.md)**.

---

## Table of Contents

1. [Highlights](#highlights)
2. [Architecture](#architecture)
3. [Repository Layout](#repository-layout)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [CLI Reference](#cli-reference)
7. [The Full Pipeline](#the-full-pipeline-end-to-end)
8. [Design & Engineering Notes](#design--engineering-notes)
9. [Tests](#tests)
10. [Knowledge Map](#knowledge-map)
11. [Limitations & Future Work](#limitations--future-work)

---

## Highlights

| Capability                         | What you get                                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Multi-backend inference**        | One `infer()` call across `pytorch`, `onnx_fp32`, `onnx_int8` — PyTorch / ONNX / INT8 share the same pre/post. |
| **YOLOv8 → ONNX export**           | opset 17, dynamic shape by default, ONNX simplifier, runtime-validated session.                                |
| **Static INT8 PTQ (QDQ)**          | QUInt8 activations, QInt8 symmetric per-channel weights, `MinMax` or `Entropy` (KL) calibration.               |
| **Head-aware quantization policy** | The **entire Detect head (`/model.22/`) kept FP32 by name-prefix**; op-type `Sigmoid/Softmax` exclusion as residual fallback. Op-type-only collapses cls scores to 0. |
| **Diverse calibration sampler**    | `CalibrationSampler`: sqrt-frequency stratified → phash dedup → ResNet-50 farthest-first traversal.            |
| **Two consistency modes**          | `tensor` (strict `assert_allclose`) for export regressions, `detection` (IoU / class / score) for INT8 regressions.|
| **Throughput vs. accuracy**        | p50 / p95 / p99 latency, FPS, peak RSS, CUDA peak memory, full-set **per-class mAP** via Ultralytics.          |
| **CUDA IO Binding via DLPack**     | Zero-copy input from `torch.Tensor` → `OrtValue` → CUDA tensors; CPU path is numpy.                            |
| **Operational hygiene**            | Logging w/ rotation, atomic JSON writes, `malloc_trim` between backend runs (Linux/glibc only).               |

---

## Architecture

```text
┌──────────────────────── yolov8s_ort CLI (main.py) ─────────────────────────┐
│                                                                            │
│   export │ inspect │ quantize │ infer │ consistency │ benchmark            │
│                                                                            │
└────┬────────────┬───────────┬──────────┬───────────┬──────────┬────────────┘
     │            │           │          │           │          │
   cli/*       utils/      src/         src/        src/        src/
   (argparse   (logging,   (export.py,  (engine.py,  (consistency,benchmark.py
    + thin      I/O,       quantize.py, preprocess.py           etc.)
    wrappers)   metadata,  postprocess.py, sampler.py
                CPU        engine.py,
                threading) benchmark.py,
                           consistency.py
```

* **`cli/`** — thin argparse wrappers; no business logic. Each `cli/<cmd>.py` exposes `add_parser()` and `run()`; configs are typed dataclasses in `cli/schema.py`.
* **`utils/`** — logging (rotating file + console), path validation, image discovery, visualization, ONNX metadata + SHA-256, percentiles, CPU-thread configuration.
* **`src/`** — the real machinery: preprocessing (letterbox, batched), postprocessing (NMS via Ultralytics + scale-back), unified inference engine, INT8 quantization pipeline, consistency harness, benchmark driver, calibration sampler.
* **`tests/`** — pure-Python unit tests for post-processing, comparison helpers, utils; intentionally model-free so they run on any laptop.

---

## Repository Layout

```
edge-ai-deployment/
├── main.py                    # CLI entry: dispatches to cli/*
├── clean.sh                   # Reset: drop *.onnx, *.cache, *.tmp, Python/pytest caches, logs/, results/, runs/
├── ARTIFACTS.md               # Where models/ + data/ come from: Release assets, sha256-pinned
├── pyproject.toml             # black / isort / pytest config
├── .flake8                    # flake8 config (flake8 7.x does not read pyproject.toml)
├── .gitignore                 # models/, data/, results/, logs/, runs/ stay out of git
├── requirements.txt           # Core runtime deps (shared CPU/GPU; no torch/ORT) + dev tools
├── requirements-cpu.txt       # CPU overlay: +cpu torch/torchvision wheels, onnxruntime
├── requirements-kaggle.txt    # Kaggle T4 overlay: no torch reinstall, no numpy pin
│
├── .github/workflows/ci.yml   # CI: flake8 + pytest on Python 3.11 / 3.12
│
├── docs/                      # Deep-dive documentation
│   ├── ARCHITECTURE.md        # Architecture deep-dive: pipeline, quantization, benchmarking
│   └── TRAINING.md            # Training narrative: subset build, analysis, staged long-tail
│
├── train/                     # Training upstream: COCO 12-class subset build + YOLOv8s training
│
├── cli/                       # CLI wrappers + shared dataclass configs
│   ├── __init__.py            # Re-exports *Config dataclasses
│   ├── schema.py              # ExportConfig / QuantizeConfig / InferConfig / ...
│   ├── export.py              # `export`  — PyTorch → ONNX FP32
│   ├── inspect.py             # `inspect` — metadata (path, sha256, opset, I/O)
│   ├── quantize.py            # `quantize`— ONNX FP32 → INT8 (QDQ)
│   ├── infer.py               # `infer`   — batch inference + visualization
│   ├── consistency.py         # `consistency` — two models, tensor or detection mode
│   └── benchmark.py           # `benchmark` — p50/p95/p99, FPS, RSS, mAP
│
├── src/                       # Domain logic
│   ├── __init__.py
│   ├── _version.py            # Single-source __version__ (src / cli / utils / main)
│   ├── export.py              # ONNX export + structural/runtime validation
│   ├── preprocess.py          # letterbox + ThreadPoolExecutor batcher
│   ├── postprocess.py         # NMS via Ultralytics + scale_boxes + clip
│   ├── engine.py              # Unified YOLOv8Engine (PT / ONNX / INT8 × CPU / CUDA)
│   ├── quantize.py            # YOLOv8CalibrationDataReader + quantize_static
│   ├── sampler.py             # CalibrationSampler: stratified + phash + farthest-first
│   ├── consistency.py         # ModelWrapper + tensor & detection comparison
│   └── benchmark.py           # Benchmark class: latency / throughput / memory / mAP
│
├── utils/                     # Cross-cutting helpers
│   ├── __init__.py
│   ├── logging.py             # Rotating file handler, idempotent root config
│   ├── common.py              # existing_validation, load_images
│   ├── comparison.py          # Pure-NumPy tensor & detection comparison (no ultralytics)
│   ├── model_utils.py         # sha256, file_size_mb, inspect_onnx, compare_models
│   ├── metrics.py             # percentiles, summarize_runs
│   ├── threading.py           # clamp_workers (ThreadPoolExecutor cap = 2×CPU)
│   └── visualization.py       # draw_detections / save_annotated_image
│
└── tests/                     # Pure-Python unit tests
    ├── conftest.py
    ├── test_postprocess.py    # Tensor / ndarray / list inputs, scale-back, clips
    ├── test_consistency.py    # IoU, compare_tensors, compare_detections, purity guard
    ├── test_quantize.py       # Calibration-reader regressions (_filter_readable, all-unreadable)
    ├── test_utils.py          # sha256, percentiles, threading, viz, compare_models
    └── test_train_staged_retry.py  # Training retry/resume state machine (self-contained)
```

---

## Installation

```bash
# 1. Create and activate a fresh Python 3.11 / 3.12 environment
python -m venv .venv
source .venv/bin/activate            # PowerShell:  .venv\Scripts\activate

# 2. Install dependencies — pick the file matching your runtime:
#    CPU (local):    pip install -r requirements-cpu.txt
#    GPU (Kaggle T4 / any env with a preinstalled CUDA torch):
#                    pip install -r requirements-kaggle.txt
#
#    Do NOT use plain `requirements.txt` alone — it omits torch / torchvision / onnxruntime (they live in the runtime variants) and `import torch` / `import onnxruntime` will fail.
pip install -r requirements-cpu.txt
```

> **CUDA users:** install the `onnxruntime-gpu` package matching your CUDA toolkit and the matching PyTorch wheel. ONNX Runtime, PyTorch, and CUDA versions must agree.

The `models/` directory must contain:

* `models/yolov8s.pt`               — Ultralytics YOLOv8s PyTorch checkpoint
* `models/yolov8s_fp32.onnx`        — produced by `python main.py export …`
* `models/yolov8s_int8.onnx`        — produced by `python main.py quantize …`
* `models/resnet50-11ad3fa6.pth`    — *optional*: only used if you pass `--resnet50` explicitly. By default the sampler auto-downloads torchvision's pretrained weights into torchvision's own hub cache (not `models/`).

The `data/` directory must contain a YOLO-format dataset with `data.yaml`.

> **Getting the artifacts:** neither `models/yolov8s.pt` nor `data/` is in git —
> both directories are git-ignored. **[ARTIFACTS.md](ARTIFACTS.md)** documents the
> GitHub Release assets, their sha256 pins, and the download + verify commands
> that populate the two directories above.

---

## Quick Start

```bash
# 1. Export a PyTorch YOLOv8s checkpoint to ONNX FP32
python main.py export --model models/yolov8s.pt --output models/yolov8s_fp32.onnx --imgsz 640 --opset 17

# 2. Inspect metadata (size, sha256, opset, I/O shapes)
python main.py inspect --model models/yolov8s_fp32.onnx

# 3. Compare the PT export against the PT-ONNX export
python main.py consistency --model1 models/yolov8s.pt --model2 models/yolov8s_fp32.onnx --imgs-input data --mode tensor

# 4. Quantize FP32 → INT8 (QDQ, MinMax by default)
python main.py quantize --model models/yolov8s_fp32.onnx --output models/yolov8s_int8.onnx --imgs-input data --max-cal-samples 300 --method MinMax

# 5. Run inference on images and draw boxes
python main.py infer --backend onnx_int8 --model models/yolov8s_int8.onnx --imgs-input data/images/val --output-dir results/predictions/onnx_int8

# 6. Benchmark all three backends, with optional mAP validation
python main.py benchmark --model pytorch:models/yolov8s.pt --model onnx_fp32:models/yolov8s_fp32.onnx --model onnx_int8:models/yolov8s_int8.onnx --imgs-input data --validation --warmup 10 --runs 25
```

---

## CLI Reference

| Command       | Purpose                                                                 | Key flags (default in **bold**)                                                                 |
| ------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `export`      | PyTorch `.pt` → ONNX FP32 (opset **17**, dynamic, onnxslim simplify).    | `--model` (req) · `--output` · `--imgsz 640` · `--opset 17` · `--no-dynamic` · `--no-simplify` · `--no-validate` · `--nms` · `--device cpu` |
| `inspect`     | Print ONNX metadata (sha256, opset, I/O, size, producer).               | `--model` (req) · `--compare`                                                                   |
| `quantize`    | ONNX FP32 → INT8 (QDQ, **MinMax** or **Entropy** calibration).          | `--model` · `--output` · `--imgs-input` · `--imgsz 640` · `--batch-size 1` · `--method MinMax` · `--max-cal-samples 300` · `--resnet50` · `--device cpu` |
| `infer`       | End-to-end inference with annotated overlays.                           | `--backend onnx_fp32` · `--model` · `--imgs-input` (req) · `--output-dir` · `--imgsz 640` · `--batch-size 8` · `--conf 0.25` · `--iou 0.45` · `--max-det 300` · `--no-save` |
| `consistency` | Two-model comparison (`tensor` strict · `detection` relaxed).           | `--model1` · `--model2` · `--imgs-input` · `--mode tensor` · `--max-images 20` · `--img-sizes` · `--batch-sizes` · `--atol 1e-3` · `--rtol 1e-2` · `--report-path` |
| `benchmark`   | Latency / throughput / memory; optional **mAP** validation.             | `--model backend:path` (repeatable) · `--imgs-input` · `--max-images 16` · `--warmup 10` · `--runs 25` · `--validation` · `--conf-threshold 0.001` · `--iou-threshold 0.7` |

---

## The Full Pipeline (end-to-end)

```
yolov8s.pt
   │  Ultralytics export, opset 17, simplify, dynamic shapes
   ▼
yolov8s_fp32.onnx
   │  inspect: print sha256 / size / I/O shapes
   │  quant_pre_process: shape inference + baseline graph optimization
   │  CalibrationSampler: stratified → phash-dedup → farthest-first
   │  YOLOv8CalibrationDataReader: yields {input: float32 batch}
   │  quantize_static: QDQ, QUInt8 acts, QInt8 symmetric per-channel weights
   │  CalibrationMethod {MinMax | Entropy (KL-divergence)}
   ▼
yolov8s_int8.onnx
   │  consistency PT vs FP32 (mode=tensor)
   │  consistency FP32 vs INT8 (mode=detection) ← per-image IoU/class/score
   ▼
YOLOv8Engine
   ├─ CPU     : numpy float32 in/out
   └─ CUDA    : IO Binding w/ DLPack — OrtValue stays on the device
   │
   │  preprocess_imgs   (thread pool)
   │      ↓
   │  session.run() or session.run_with_iobinding()
   │      ↓
   │  post_process        (NMS via Ultralytics, scale_boxes via letterbox r/p)
   │      ↓
   │  results: List[List[(x1,y1,x2,y2,conf,cls)]]
   ▼
Benchmark
   ├─ latency:    mean / p50 / p95 / p99 per run
   ├─ throughput: FPS = images / mean_total_s
   ├─ memory:     RSS Δ (CPU) or torch.cuda.max_memory_allocated (CUDA)
   └─ (optional) Ultralytics val → per-class mAP50 / mAP50-95 + mean
```

---

## Design & Engineering Notes

The non-obvious decisions that distinguish a research script from a deployable toolkit. Each is documented **once** — with code refs, rationale, and measured numbers — in the [architecture deep-dive](docs/ARCHITECTURE.md):

| Decision | One-liner | Deep-dive |
| -------- | --------- | --------- |
| Per-backend warm-up | Warm-up runs real inference through the *deployed* path (IO Binding + DLPack on CUDA), not `session.run` on host numpy. | [§4.3](docs/ARCHITECTURE.md#43-cuda-path--io-binding-via-dlpack) |
| CUDA IO Binding via DLPack | Zero host→device copy on input; the engine additionally defers the device→host output copy. Three entry points by design. | [§4.3](docs/ARCHITECTURE.md#43-cuda-path--io-binding-via-dlpack) |
| INT8 QDQ + whole-Detect-head FP32 | QDQ maximizes kernel fusion; the `/model.22/` head stays FP32 by name-prefix (op-type-only exclusion collapses cls scores). | [§5.5](docs/ARCHITECTURE.md#55-keep-the-entire-detect-head-in-fp32) |
| Calibration sampling | sqrt-frequency stratification → phash dedup → ResNet-50 farthest-first: random sampling drowns rare classes, duplicates waste budget. | [§6](docs/ARCHITECTURE.md#6-calibration-sampler-design) |
| Two consistency modes | `tensor` for export regressions; `detection` (per-image IoU/class/score gates authoritative) for INT8. Raw-output p99 is advisory only. | [§7.1](docs/ARCHITECTURE.md#71-two-models-two-modes) |
| CPU thread tuning | `cv2=0`, `torch=min(4,ncpu)`, ORT `intra=min(4,ncpu)/inter=2` — ORT's "all cores" default oversubscribes single-stream inference. | [§9](docs/ARCHITECTURE.md#9-threading-memory-and-logging-hygiene) |
| Memory hygiene between runs | `gc.collect()` + `malloc_trim(0)` (Linux) + CUDA cache/peak reset so `rss_increase_mb` reflects only the current backend. | [§8.2](docs/ARCHITECTURE.md#82-memory-hygiene-between-backends) |
| Atomic + recoverable JSON | `tmp + os.replace`, re-dumped after every (imgsz, batch-size) combination — a SIGKILL never leaves a half-written report. | [§7.4](docs/ARCHITECTURE.md#74-atomic-recoverable-report-writing) |
| Logger hygiene | Rotating file handler + stdout, third-party logs raised to WARNING, idempotent init. | [§9](docs/ARCHITECTURE.md#9-threading-memory-and-logging-hygiene) |

---

## Tests

```bash
pytest -q
```

Pure-Python unit tests; **no model or GPU required**:

| Test file                       | Coverage                                                                                                |
| ------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `tests/test_postprocess.py`     | `ndarray` / `tensor` / `list` inputs, scale-back, clipping, conf filtering, output types and rounding, layout-heuristic regressions. |
| `tests/test_consistency.py`     | `compute_iou`, `compare_tensors` (tensor / detection modes, NaN detection), `compare_detections`, re-export identity, `consistency` parser defaults (`--report-path`), ultralytics-free purity guard. |
| `tests/test_quantize.py`        | Calibration-reader regressions: `_filter_readable` keep/drop behavior, all-unreadable raise.             |
| `tests/test_utils.py`           | `cosine_similarity`, `percentiles`/`summarize_runs`, `color_for_class`, `draw_detections`, `existing_validation` (+ `PathValidationError` ARTIFACTS.md hint), `resolve_model_arg`/`onnx_only` path validation, `sha256_of_file`, `file_size_mb`, `clamp_workers`, `compare_models`, cross-package `__version__` consistency. |
| `tests/test_train_staged_retry.py` | Training-side retry/resume state machine — self-contained, no ultralytics.                            |

`tests/conftest.py` adds the project root to `sys.path` and skips the `data_dir` fixture when `data/` is absent (so unit tests don't require the full dataset).

---

## Knowledge Map

This is the conceptual inventory the codebase exercises — most items map to a concrete implementation in this repo; the mechanics are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Model formats & graph surgery
* ONNX, ONNX opset (11 / 13 / 17), IR version, dynamic axes
* PyTorch → ONNX tracing (`onnx.export`), onnxslim simplification, `quant_pre_process`
* ONNX graph representation: nodes, initializers, value_info, opset_import
* QDQ vs QOperator formats; Conv + Q/DQ fusion

### Quantization (PTQ)
* Static vs dynamic PTQ; calibration set design
* QUInt8 (asymmetric) activations vs QInt8 (symmetric) weights — why the mix?
* Per-channel vs per-tensor weight quantization; when per-channel saves accuracy
* CalibrationMethod.MinMax vs Entropy (KL-divergence)
* Whole-Detect-head FP32 protection (`/model.22/` by name prefix) — and why op-type-only is not enough (cls scores collapse to 0)

### Inference runtimes
* ONNX Runtime providers: CPU, CUDA, TensorRT, OpenVINO, DirectML, NNAPI, CoreML
* IO Binding: `OrtValue`, `bind_ortvalue_input`, `bind_output("cuda")`
* DLPack: zero-copy bridge between PyTorch CUDA tensors and ORT CUDA tensors
* ORT SessionOptions: `graph_optimization_level=ORT_ENABLE_ALL`, `intra_op_num_threads`, `inter_op_num_threads`
* `arena_extend_strategy=kNextPowerOfTwo`, `cudnn_conv_algo_search=HEURISTIC`

### YOLOv8 model & post-processing
* YOLOv8 architecture: backbone (C2f / SPPF) + PAN-FPN neck + Detect head
* Detect head outputs: `bbox` (4) + `cls` (nc) — anchor-free, decoupled
* Letterbox preprocessing: aspect-preserving resize + 114-grey padding
* NMS via `ultralytics.utils.nms.non_max_suppression`
* `scale_boxes(img1_shape, boxes, img0_shape, ratio_pad)` to map back to original image coordinates

### Performance engineering
* Tail latency (p50/p90/p95/p99) vs mean — production matters at p95+
* `time.perf_counter` for sub-microsecond timing; `torch.cuda.synchronize` before measuring CUDA runtimes
* `malloc_trim(0)` and `torch.cuda.empty_cache()` between runs for clean memory baselines
* `psutil.Process().memory_info().rss` for CPU peak
* `torch.cuda.max_memory_allocated` for CUDA peak

### Software engineering
* Argparse subcommand pattern: each `cli/<cmd>.py` exposes `add_parser()` + `run(args)`; configs as `@dataclass` from `cli/schema.py`
* Threading: OpenCV internal threads, `ThreadPoolExecutor` for letterbox, intra/inter op thread counts
* Logging: rotating file handler + stdout, format strings, third-party log silencing, idempotent `setup_logging(force=...)`
* Atomic JSON writes for crash-safe reports
* Unit-testing a model-heavy codebase without a model: pure-Python, no I/O

---

## Limitations & Future Work

* **No TensorRT engine export in this repo** — out of scope. The architecture is designed so a future `cli/trtexport.py` can plug into `engine.py` the same way `pytorch` and `onnx_*` do.
* **Static INT8 only.** Dynamic quantization and QAT would each deserve a separate pipeline; the head-exclusion (whole `/model.22/` FP32) would likely need re-tuning for QAT-trained models.
* **CPU-only measurement bias.** The benchmark records RSS correctly on CPU but doesn't differentiate anonymous vs file-backed pages. On Linux the `malloc_trim` workaround is good enough; on Windows it is a no-op (the script silently skips).
* **YOLOv8s 12-class COCO subset only.** The post-processing layout heuristic (permute to `(bs, 4+nc, N)` when `shape[1] > shape[2] and shape[1] > SMALL`) assumes `num_boxes >> 4+nc` — always true for YOLOv8 at 640px (`N=8400`, `4+nc=16`). It cannot disambiguate a model that genuinely outputs `(bs, 4+nc, N)` with `4+nc > N`; `post_process(nc=...)` already implements the exact-dim resolution for that case — the engine/benchmark call sites just need to thread `nc` in when it shows up.
* **No model artifacts in VCS.** Tests are deliberately model-free; CI (`.github/workflows/ci.yml`) runs the same `flake8 .` + `pytest -q` gate on Python 3.11 / 3.12 for every push and pull request targeting `main`. Obtain the model/data artifacts per [ARTIFACTS.md](ARTIFACTS.md); regenerate derived artifacts locally via the [Quick Start](#quick-start) commands.
