# YOLOv8s ONNX Runtime Toolchain

A toolchain for taking a **YOLOv8s** PyTorch checkpoint through the
edge-deployment lifecycle: export → ONNX Runtime inference → static INT8
quantization (QDQ) → consistency & accuracy validation → latency / throughput /
memory / mAP benchmark, designed to be reproducible from a single CLI. The CLI
layer is Python; a C++ inference API is planned alongside the remaining deploy
milestones.

The deploy half lands in milestones; the currently reproducible commands are
**`export`** and **`inspect`**. The training half — COCO 12-class subset build,
dataset analysis, and baseline vs. staged long-tail training that produces
`models/yolov8s.pt` — is complete and lives in [`train/`](train/), documented in
**[docs/TRAINING.md](docs/TRAINING.md)**.

---

## Table of Contents

1. [Highlights](#highlights)
2. [Architecture](#architecture)
3. [Repository Layout](#repository-layout)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [CLI Reference](#cli-reference)
7. [Tests](#tests)
8. [Limitations & Future Work](#limitations--future-work)

---

## Highlights

| Capability                  | Implementation                                                                                                                                            |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **YOLOv8 → ONNX export**    | opset 17, dynamic shape by default, ONNX simplifier; runtime-validated session (structural check, shape inference, ORT probe incl. dynamic-batch check).  |
| **ONNX model inspection**   | sha256, file size, opset, I/O names and shapes; optional two-model side-by-side comparison.                                                               |
| **Path validation**         | `existing_validation` + `PathValidationError` (an `argparse.ArgumentTypeError`, so argparse prints the message verbatim); missing paths under `models/` or `data/` point at [ARTIFACTS.md](ARTIFACTS.md). |
| **Lazy imports**            | `src` resolves domain attributes via PEP 562 `__getattr__`; torch / ultralytics stay deferred — CLI startup and `--help` load only the argparse + utils layer (which itself imports onnx / onnxruntime / cv2).                  |
| **Operational hygiene**     | Rotating file + console logging, third-party logs raised to WARNING, per-module logger names.                                                             |

---

## Architecture

```text
main.py  —  command registry + logging setup
  └── cli/<cmd>.py   thin argparse wrappers: add_parser() + run(args)
        └── cli/schema.py   typed @dataclass configs
              └── src/      domain logic: ONNX export + runtime validation
  utils/   cross-cutting helpers: logging, path validation, I/O, metadata,
           CPU threading, percentiles  (imported from cli/ and src/)
  tests/   unit tests (no model / GPU required)
```

* **`cli/`** — thin argparse wrappers; no business logic. Each `cli/<cmd>.py` exposes `add_parser()`
  and `run()`; configs are typed dataclasses in `cli/schema.py`. Path defaults are centralized in
  `cli/__init__.py`. The current commands validate paths at parse time via argparse `type=`
  callables (`existing_validation`, `onnx_only`).
* **`src/`** — domain logic: ONNX export with structural validation (onnx.checker, shape
  inference) and a runtime probe through ONNX Runtime.
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
│
├── docs/
│   └── TRAINING.md            # Training docs: subset build, analysis, staged long-tail
│
├── train/                     # Training upstream: COCO 12-class subset build + YOLOv8s training
│
├── cli/                       # CLI wrappers + shared dataclass configs
│   ├── __init__.py            # Shared path defaults
│   ├── schema.py              # ExportConfig dataclass
│   ├── export.py              # `export`  — PyTorch → ONNX FP32
│   └── inspect.py             # `inspect` — metadata (path, sha256, opset, I/O)
│
├── src/                       # Domain logic
│   ├── __init__.py            # PEP 562 lazy re-exports
│   ├── _version.py            # Single-source __version__ (src / cli / utils / main)
│   └── export.py              # ONNX export + structural/runtime validation
│
├── utils/                     # Cross-cutting helpers
│   ├── __init__.py
│   ├── logging.py             # Rotating file handler, idempotent root config
│   ├── common.py              # existing_validation, load_images
│   ├── comparison.py          # Pure-NumPy tensor & detection comparison (no ultralytics)
│   ├── model_utils.py         # OnnxMetadata, sha256, file_size_mb, inspect_onnx, compare_models, select_providers
│   ├── metrics.py             # percentiles, summarize_runs
│   ├── threading.py           # clamp_workers (ThreadPoolExecutor cap = 2×CPU)
│   └── visualization.py       # draw_detections / save_annotated_image
│
└── tests/                     # Unit tests (no model / GPU required)
    ├── conftest.py            # Adds project root to sys.path; data_dir fixture skips without data/
    ├── test_utils.py          # utils + CLI path-validation regressions
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
> PyTorch wheel. ONNX Runtime, PyTorch, and CUDA versions must agree.

The `models/` directory must contain:

* `models/yolov8s.pt`               — Ultralytics YOLOv8s PyTorch checkpoint (downloaded; see the note below)

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
```

---

## CLI Reference

| Command  | Purpose                                                    | Key flags (default in **bold**)                                                                                                            |
| -------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `export` | PyTorch `.pt` → ONNX FP32 (opset **17**, dynamic, onnxslim simplify). | `--model` (req) · `--output` · `--imgsz 640` · `--opset 17` · `--no-dynamic` · `--no-simplify` · `--no-validate` · `--nms` · `--device cpu` |
| `inspect`| Print ONNX metadata (sha256, opset, I/O, size, producer).  | `--model` (req) · `--compare`                                                                                                              |

---

## Tests

```bash
pytest -q
```

Unit tests; **no model or GPU required**:

| Test file             | Coverage                                                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `tests/test_utils.py` | `cosine_similarity`, `percentiles`/`summarize_runs`, `color_for_class`, `draw_detections`, `existing_validation` (+ `PathValidationError` ARTIFACTS.md hint), `onnx_only` path validation, `sha256_of_file`, `file_size_mb`, `clamp_workers`, `compare_models`, cross-package `__version__` consistency. |
| `tests/test_train_staged_retry.py` | Staged-training retry/resume state machine (`train/train_staged/utils.py`, loaded by file path). Imports numpy/torch/yaml only; no ultralytics. |

`tests/conftest.py` adds the project root to `sys.path` and skips the `data_dir` fixture when
`data/` is absent (so unit tests don't require the full dataset).

---

## Limitations & Future Work

* **Deploy-half milestones land incrementally.** The remaining commands —
  `quantize` (static INT8 PTQ), `infer`, `consistency`, `benchmark`, and the
  OpenVINO / TensorRT backends — enter the CLI with their milestones; the
  intro paragraph states the target lifecycle. A C++ inference API follows
  once the ONNX Runtime path is complete.
* **CPU-only measurement bias (planned).** The upcoming `benchmark` milestone
  will record process RSS on CPU; anonymous vs file-backed pages will not be
  differentiated.
* **No model artifacts in VCS.** Tests are model-free. Obtain the model/data
  artifacts per [ARTIFACTS.md](ARTIFACTS.md); regenerate derived artifacts
  locally via the [Quick Start](#quick-start) commands.
