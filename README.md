# YOLOv8s ONNX Runtime Toolchain

A toolchain for taking a **YOLOv8s** PyTorch checkpoint through the
edge-deployment lifecycle: export → ONNX Runtime inference → static INT8
quantization (QDQ) → consistency & accuracy validation → latency / throughput /
memory / mAP benchmark, designed to be reproducible from a single CLI. The CLI
layer is Python; a C++ inference API is planned alongside the remaining deploy
milestones.

The deploy half lands in milestones; the currently reproducible commands are
**`export`**, **`inspect`**, **`quantize`**, **`infer`**, **`consistency`**,
and **`benchmark`**.
The training half — COCO 12-class subset build, dataset analysis, and baseline
vs. staged long-tail training that produces `models/yolov8s.pt` — is complete
and lives in [`train/`](train/), documented in
**[docs/TRAINING.md](docs/TRAINING.md)**.

---

## Table of Contents

1. [Highlights](#highlights)
2. [Architecture](#architecture)
3. [Repository Layout](#repository-layout)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [CLI Reference](#cli-reference)
7. [The Full Pipeline](#the-full-pipeline-end-to-end)
8. [Tests](#tests)
9. [Limitations & Future Work](#limitations--future-work)

---

## Highlights

| Capability                  | Implementation                                                                                                                                            |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Multi-backend inference** | `YOLOv8Engine`: one `infer()` API across `pytorch`, `onnx_fp32`, `onnx_int8` × `cpu`/`cuda` — all backends share the same preprocess / post-process pipeline; the `infer` CLI and a `preprocess_imgs`-backed batch loop sit on top. |
| **YOLOv8 → ONNX export**    | opset 17, dynamic shape by default, ONNX simplifier; runtime-validated session (structural check, shape inference, ORT probe incl. dynamic-batch check).  |
| **ONNX model inspection**   | sha256, file size, opset, I/O names and shapes; optional two-model side-by-side comparison.                                                               |
| **Static INT8 PTQ (QDQ)**   | `quant_pre_process` → `quantize_static`: QUInt8 asymmetric activations, QInt8 symmetric per-channel weights, `MinMax` or `Entropy` (KL-divergence) calibration. |
| **Head-aware quantization policy** | The entire Detect head (`/model.22/`) is kept in FP32 via name-prefix exclusion, resolved to concrete node names from the pre-processed graph; op-type `Sigmoid`/`Softmax` exclusion is the fallback layer for builds that rename head nodes. Quantizing the head cls-logit branch collapsed the post-Sigmoid class scores to zero in the validated YOLOv8s configuration. |
| **Diverse calibration sampler** | `CalibrationSampler`: sqrt-frequency stratified class quotas (floor 1, greedy trim back to budget) → perceptual-hash dedup → ResNet-50 farthest-first traversal over L2-normalized features. |
| **Two consistency modes** | `tensor` (strict `assert_allclose`) for export regressions; `detection` (per-image IoU / class / score gates) for INT8 regressions, with the raw-tensor stats advisory. Report written atomically after every (imgsz, batch) config; failed-sample images copied out; exit code 1 on FAIL. |
| **Throughput vs. accuracy** | `benchmark`: p50 / p90 / p95 / p99 latency, FPS, peak RSS (CUDA allocator peak on `--device cuda`), plus optional full-set **per-class mAP50 / mAP50-95** via a backend-agnostic native COCO evaluator (`utils/map_eval.py`) driven over each backend's `infer()`; a `mAP_source` column records per-row provenance (native evaluator / unsupported / failed / no GT / disabled). |
| **Batched preprocessing**   | `preprocess_imgs`: letterbox (aspect-ratio-preserving resize + pad) in a `ThreadPoolExecutor`, BGR→RGB, HWC→CHW, /255 → float32 NCHW batch tensors; the INT8 calibration reader consumes it batch by batch. `preprocess_frames` takes already-decoded BGR frames through the same per-image core and batch assembly. |
| **Post-processing**         | `post_process`: NMS via Ultralytics `non_max_suppression`, letterbox→original scale-back via `scale_boxes`, per-image clipping, `[(x1, y1, x2, y2, conf, cls), …]` tuples in original-image coordinates; accepts `np.ndarray` / `torch.Tensor` / list outputs in either `(bs, 4+nc, N)` or `(bs, N, 4+nc)` layout. |
| **CUDA IO Binding via DLPack** | On CUDA the engine binds the input `torch.Tensor` as an on-device `OrtValue` via DLPack (zero host→device copy) and copies outputs back once per inference; the CPU path is plain numpy `session.run`. Warm-up runs the deployed per-backend/per-device path. |
| **Path validation**         | `existing_validation` + `PathValidationError` (an `argparse.ArgumentTypeError`, so argparse prints the message verbatim); missing paths under `models/` or `data/` point at [ARTIFACTS.md](ARTIFACTS.md). |
| **Lazy domain imports**      | `src` resolves domain attributes via PEP 562 `__getattr__`; a bare `import src` stays dependency-light — submodules load on first attribute access. `export` defers its domain import into `run()`; `quantize`, `infer`, `consistency`, and `benchmark` bind at module level, so `main.py` startup loads torch / torchvision / onnx / onnxruntime / cv2 / imagehash / ultralytics / pandas / psutil. |
| **Operational hygiene**     | Rotating file + console logging, third-party logs raised to WARNING, per-module logger names; `gc.collect()` + `malloc_trim(0)` (Linux/glibc only) between benchmark backends so the RSS delta reflects only the current backend. |

---

## Architecture

```text
main.py  —  command registry + logging setup
  └── cli/<cmd>.py   thin argparse wrappers: add_parser() + run(args)
        └── cli/schema.py   typed @dataclass configs
              └── src/      domain logic: ONNX export + validation,
                            letterbox preprocessing, post-processing,
                            unified inference engine, INT8 quantization,
                            calibration sampler, two-model consistency
                            validation, performance benchmark
  utils/   cross-cutting helpers: logging, path validation, I/O, metadata,
           CPU threading, percentiles  (imported from cli/ and src/)
  tests/   unit tests (no model / GPU required)
```

* **`cli/`** — thin argparse wrappers; no business logic. Each `cli/<cmd>.py` exposes `add_parser()`
  and `run()`; configs are typed dataclasses in `cli/schema.py`. Path defaults are centralized in
  `cli/__init__.py`. Required-path commands validate at parse time via argparse `type=`
  callables (`existing_validation`, `onnx_only`); commands with optional path flags
  (`consistency`) declare `default=None` and resolve the documented default at `run()` time
  via `resolve_path_arg`.
* **`src/`** — domain logic: ONNX export with structural validation (onnx.checker, shape
  inference) and a runtime probe through ONNX Runtime; letterbox preprocessing with a
  thread-pool batcher; post-processing (NMS via Ultralytics + scale-back to original
  coordinates); a unified inference engine (`YOLOv8Engine`: pytorch / onnx_fp32 / onnx_int8 ×
  cpu / cuda); static INT8 quantization (calibration reader over `preprocess_imgs` +
  `quantize_static`); calibration sampler (stratified → phash dedup → farthest-first);
  two-model consistency validation (`ModelWrapper` over PT / ONNX, tensor and
  detection comparison from `utils.comparison`); a performance benchmark
  (`Benchmark`: per-backend timed loops over the shared preprocess → forward
  → post-process pipeline, percentile latency, FPS, peak memory, optional
  native mAP validation).
* **`utils/`** — logging (rotating file + console), path validation, image discovery, visualization,
  ONNX metadata + SHA-256, percentiles, CPU-thread configuration.
* **`tests/`** — unit tests; model-free so they run on any laptop (`test_train_staged_retry.py`
  additionally imports torch).

---

## Repository Layout

```
edge-ai-deployment/
├── main.py                    # CLI entry: dispatches to cli/*
├── clean.sh                   # Reset: drop *.onnx, *.cache, *.tmp, caches, logs/, results/, runs/
├── ARTIFACTS.md               # Where models/ + data/ come from: Release assets, sha256-pinned
├── pyproject.toml             # black / isort / pytest config
├── .flake8                    # flake8 config (flake8 7.x does not read pyproject.toml)
├── .gitignore                 # models/, data/, coco12-val.zip, results/, runs/, logs/ stay out of git
├── requirements.txt           # Core runtime deps (shared CPU/GPU; no torch/ORT) + dev tools
├── requirements-cpu.txt       # CPU overlay: +cpu torch/torchvision wheels, onnxruntime
├── requirements-kaggle.txt    # GPU overlay: keeps preinstalled CUDA torch; onnxruntime-gpu
│
├── docs/
│   └── TRAINING.md            # Training docs: subset build, analysis, staged long-tail
│
├── train/                     # Training upstream: COCO 12-class subset build + YOLOv8s training
│
├── cli/                       # CLI wrappers + shared dataclass configs
│   ├── __init__.py            # Shared path defaults, resolve_path_arg, run_timestamp
│   ├── schema.py              # ExportConfig / QuantizeConfig / InferConfig / ...
│   ├── export.py              # `export`  — PyTorch → ONNX FP32
│   ├── inspect.py             # `inspect` — metadata (path, sha256, opset, I/O)
│   ├── quantize.py            # `quantize`— ONNX FP32 → INT8 (QDQ)
│   ├── infer.py               # `infer`   — batch inference + visualization
│   ├── consistency.py         # `consistency` — two models, tensor or detection mode
│   └── benchmark.py           # `benchmark` — p50/p90/p95/p99, FPS, RSS, mAP
│
├── src/                       # Domain logic
│   ├── __init__.py            # PEP 562 lazy re-exports
│   ├── _version.py            # Single-source __version__ (src / cli / utils / main)
│   ├── export.py              # ONNX export + structural/runtime validation
│   ├── preprocess.py          # letterbox + ThreadPoolExecutor batch tensor assembly
│   ├── postprocess.py         # NMS via Ultralytics + scale_boxes + clip
│   ├── engine.py              # Unified YOLOv8Engine (PT / ONNX / INT8 × CPU / CUDA)
│   ├── quantize.py            # YOLOv8CalibrationDataReader + static QDQ quantization
│   ├── sampler.py             # CalibrationSampler: stratified + phash + farthest-first
│   ├── consistency.py         # ModelWrapper + tensor & detection comparison
│   └── benchmark.py           # Benchmark class: latency / throughput / memory / mAP
│
├── utils/                     # Cross-cutting helpers
│   ├── __init__.py
│   ├── logging.py             # Rotating file handler, idempotent root config
│   ├── common.py              # existing_validation, load_images
│   ├── comparison.py          # Pure-NumPy tensor & detection comparison (no ultralytics)
│   ├── model_utils.py         # OnnxMetadata, sha256, file_size_mb, inspect_onnx, compare_models, select_providers
│   ├── metrics.py             # percentiles, summarize_runs
│   ├── threading.py           # clamp_workers (ThreadPoolExecutor cap = 2×CPU)
│   ├── visualization.py       # draw_detections / save_annotated_image
│   └── map_eval.py            # Backend-agnostic native COCO mAP evaluator
│
└── tests/                     # Unit tests (no model / GPU required)
    ├── conftest.py            # Adds project root to sys.path; data_dir fixture skips without data/
    ├── test_utils.py          # utils + CLI path-validation regressions
    ├── test_postprocess.py    # Tensor / ndarray / list inputs, scale-back, clips, layout resolution
    ├── test_consistency.py    # compute_iou, compare_tensors, compare_detections, re-export identity
    ├── test_quantize.py       # Calibration-reader regressions (_filter_readable, hash-seed determinism)
    ├── test_map_eval.py       # Native mAP evaluator: IoU, COCO AP scenarios, GT loader
    ├── test_benchmark_map_plumbing.py  # benchmark --validation ↔ map_eval wiring
    └── test_train_staged_retry.py  # train_staged retry/resume state machine (loaded by file path)
```

---

## Installation

```bash
# 1. Create and activate a fresh Python 3.12 / 3.13 environment
#    (torch 2.6.0+cpu ships cp312/cp313 wheels; onnxruntime 1.26.0 needs >= 3.11)
python -m venv .venv
source .venv/bin/activate            # PowerShell:  .venv\Scripts\Activate.ps1

# 2. Install dependencies — CPU (local):
#    Do NOT use plain `requirements.txt` alone — it omits torch / torchvision /
#    onnxruntime (they live in the runtime variants) and `import torch` /
#    `import onnxruntime` will fail.
pip install -r requirements-cpu.txt
```

> **CUDA users:** install the `onnxruntime-gpu` package matching your CUDA toolkit and the matching
> PyTorch wheel. ONNX Runtime, PyTorch, and CUDA versions must agree. On images that preinstall a
> CUDA torch build (e.g. Kaggle T4), `requirements-kaggle.txt` encodes those version agreements.

The `models/` directory must contain:

* `models/yolov8s.pt`               — Ultralytics YOLOv8s PyTorch checkpoint (downloaded; see the note below)
* `models/yolov8s_fp32.onnx`        — produced by `python main.py export …`
* `models/yolov8s_int8.onnx`        — produced by `python main.py quantize …`

The `data/` directory must contain a YOLO-format dataset with `data.yaml`.

> **ResNet-50 for the calibration sampler:** `quantize`, `consistency`, and
> `benchmark` select their image sets through `CalibrationSampler`, whose
> diversity stage extracts
> features with torchvision's `resnet50`. Without `--resnet50`, torchvision
> auto-downloads `ResNet50_Weights.DEFAULT` into its own hub cache (not `models/`);
> a path passed via `--resnet50` must exist — a missing file raises
> `FileNotFoundError`.

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

# 3. Compare the PyTorch model against its ONNX FP32 export (tensor mode, strict allclose)
python main.py consistency --model1 models/yolov8s.pt --model2 models/yolov8s_fp32.onnx --imgs-input data --mode tensor --atol 1e-4 --rtol 1e-3

# 4. Quantize FP32 → INT8 (QDQ, MinMax by default)
python main.py quantize --model models/yolov8s_fp32.onnx --output models/yolov8s_int8.onnx --imgs-input data --max-cal-samples 300 --method MinMax

# 5. Compare the INT8 model against FP32 (detection mode, per-image gates)
python main.py consistency --model1 models/yolov8s_fp32.onnx --model2 models/yolov8s_int8.onnx --imgs-input data --mode detection --report-path results/consistency_detection.json

# 6. Run inference on images and save annotated overlays
python main.py infer --backend onnx_int8 --model models/yolov8s_int8.onnx --imgs-input data/images/val --output-dir results/predictions/onnx_int8

# 7. Benchmark all backends, with optional mAP validation
python main.py benchmark --model pytorch:models/yolov8s.pt --model onnx_fp32:models/yolov8s_fp32.onnx --model onnx_int8:models/yolov8s_int8.onnx --imgs-input data --validation --warmup 10 --runs 25
```

---

## CLI Reference

| Command  | Purpose                                                    | Key flags (default in **bold**)                                                                                                            |
| -------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `export` | PyTorch `.pt` → ONNX FP32 (opset **17**, dynamic, onnxslim simplify). | `--model` (req) · `--output` · `--imgsz 640` · `--opset 17` · `--no-dynamic` · `--no-simplify` · `--no-validate` · `--nms` · `--device cpu` |
| `inspect`| Print ONNX metadata (sha256, opset, I/O, size, producer).  | `--model` (req) · `--compare`                                                                                                              |
| `quantize`| ONNX FP32 → INT8 (QDQ, **MinMax** or **Entropy** calibration). | `--model` · `--output` · `--imgs-input` · `--imgsz 640` · `--batch-size 1` · `--method MinMax` · `--max-cal-samples 300` · `--resnet50` · `--device cpu` |
| `infer`   | End-to-end inference with annotated overlays. | `--backend onnx_fp32` · `--model` · `--imgs-input` (req) · `--output-dir` · `--imgsz 640` · `--batch-size 8` · `--conf 0.25` · `--iou 0.45` · `--max-det 300` · `--max-imgs 32` · `--no-save` · `--device cpu` |
| `consistency` | Two-model comparison: `tensor` (strict allclose) for export regressions · `detection` (per-image IoU / class / score gates) for INT8 regressions. JSON report + failed-sample copies; exit code 1 on FAIL. | `--model1` · `--model2` · `--imgs-input` · `--mode tensor` · `--max-images 20` · `--img-sizes 640` · `--batch-sizes 1` · `--atol 1e-3` · `--rtol 1e-2` · `--resnet50` · `--device cpu` · `--report-path results/consistency_report.json` |
| `benchmark` | Latency / throughput / memory; optional **mAP** validation (native evaluator over each backend's `infer()`; summary CSV records `mAP50` / `mAP50-95` / `mAP_source`). | `--model backend:path` (repeatable) · `--imgs-input` · `--imgsz 640` · `--batch-size 1` · `--max-images 16` · `--warmup 10` · `--runs 25` · `--validation` · `--conf-threshold 0.001` · `--iou-threshold 0.7` · `--speed-conf 0.25` · `--speed-iou 0.45` · `--resnet50` · `--device cpu` · `--no-sampler` |

---

## The Full Pipeline (end-to-end)

```text
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
   ├─ latency:    mean / p50 / p90 / p95 / p99 per run
   ├─ throughput: FPS = images / mean_total_s
   ├─ memory:     RSS Δ (CPU) or torch.cuda.max_memory_allocated (CUDA)
   └─ (optional, --validation) native COCO evaluator (utils/map_eval.py,
             driven over each backend's infer()) → per-class mAP50 /
             mAP50-95 + mean; a `mAP_source` column records per-row
             provenance (native evaluator / unsupported / failed / no GT /
             disabled)
```

---

## Tests

```bash
pytest -q
```

Unit tests; **no model or GPU required**:

| Test file             | Coverage                                                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `tests/test_postprocess.py` | `ndarray` / `tensor` / `list` inputs, scale-back, clipping, conf filtering, output types and rounding, layout-resolution regressions (`_ensure_4nc_first`: exact `nc=` mode and the magnitude heuristic). |
| `tests/test_consistency.py` | `compute_iou`, `compare_tensors` (tensor / detection modes, shape mismatch, NaN detection), `compare_detections` (empty / identical / no-match / count-diff), `src.consistency` re-export identity, `consistency` parser defaults (`--report-path`), ultralytics-free purity guard. |
| `tests/test_quantize.py` | Calibration-reader regressions: `_filter_readable` keep/drop behavior (garbage / missing / directory), all-unreadable constructor raise, stratified-sampling PYTHONHASHSEED determinism (data-gated). |
| `tests/test_map_eval.py` | Native mAP evaluator: `box_iou` (known / empty / degenerate), synthetic COCO AP scenarios with hand-computed expected values (missed GT, FP-after-TP, cross-image matching, zero-GT class exclusion), data-guarded GT loader alignment. |
| `tests/test_benchmark_map_plumbing.py` | `benchmark --validation` ↔ native evaluator wiring: `_compute_map_native` dispatch target exists; the unsupported-backend path returns without constructing any engine. |
| `tests/test_utils.py` | `cosine_similarity`, `percentiles`/`summarize_runs`, `color_for_class`, `draw_detections`, `existing_validation` (+ `PathValidationError` ARTIFACTS.md hint), `resolve_model_arg`/`onnx_only` path validation, `sha256_of_file`, `file_size_mb`, `clamp_workers`, `compare_models`, cross-package `__version__` consistency. |
| `tests/test_train_staged_retry.py` | Staged-training retry/resume state machine (`train/train_staged/utils.py`, loaded by file path). Imports numpy/torch/yaml only; no ultralytics. |

`tests/conftest.py` adds the project root to `sys.path` and skips the `data_dir` fixture when
`data/` is absent (so unit tests don't require the full dataset).

---

## Limitations & Future Work

* **Static INT8 only.** Dynamic quantization and QAT each deserve a separate
  pipeline; the head-exclusion policy (whole `/model.22/` FP32) would need
  re-tuning for QAT-trained models.
* **Deploy-half milestones land incrementally.** The remaining backends —
  OpenVINO and TensorRT — enter the CLI with their milestones; the intro
  paragraph states the target lifecycle. A C++ inference API follows once the
  ONNX Runtime path is complete.
* **YOLOv8s 12-class COCO subset only.** The post-processing layout heuristic
  (permute to `(bs, 4+nc, N)` when `shape[1] > shape[2]` and `shape[1] > SMALL`)
  assumes `num_boxes >> 4+nc` — always true for YOLOv8 at 640px (`N=8400`,
  `4+nc=16`). It cannot disambiguate a model that genuinely outputs
  `(bs, 4+nc, N)` with `4+nc > N`; `post_process(nc=...)` implements the
  exact-dim resolution for that case. The engine call sites pass `nc=`
  explicitly; the benchmark / consistency call sites use the heuristic for the
  validated configuration.
* **CPU-only measurement bias.** The benchmark records process RSS on CPU,
  sampled once after the timed loop (RSS can shrink, so it is a lower bound
  on peak RSS); anonymous vs file-backed pages are not differentiated. On
  Linux `malloc_trim(0)` is called between backend runs; on Windows it is a
  no-op.
* **No model artifacts in VCS.** Tests are model-free. Obtain the model/data
  artifacts per [ARTIFACTS.md](ARTIFACTS.md); regenerate derived artifacts
  locally via the [Quick Start](#quick-start) commands.
