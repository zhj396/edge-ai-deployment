# CLAUDE.md

This document defines the engineering constraints, architecture invariants,
validation requirements, and development workflow for this repository.

It is used by AI-assisted development tools, including Claude Code, as a
project-level engineering contract. These constraints also document the
design assumptions that should remain stable as the project evolves.

## Project Overview

`edge-ai-deployment` demonstrates an end-to-end Edge AI inference workflow:

YOLOv8s 12-class COCO subset → ONNX FP32 → static INT8 QDQ →
consistency validation → benchmarking.

The workflow is exposed through a single CLI:

```bash
python main.py {export,inspect,quantize,infer,consistency,benchmark}
```

The validated environments include:

- 13th Gen Intel Core i5-13420H CPU
- NVIDIA Tesla T4 ×2 with CUDA

The repository is designed as a reproducible technical demonstration,
with emphasis on deployment architecture, ONNX Runtime execution,
quantization, performance measurement, validation, and testability.

Repository documentation:

- `README.md` — overview, installation, quick start, CLI reference.
- `docs/ARCHITECTURE.md` — architecture deep-dive with symbol-anchored code refs.
- `ARTIFACTS.md` — where the git-ignored `models/` and `data/` artifacts come
  from (GitHub Release, sha256-pinned).
- `docs/TRAINING.md` — the training upstream that produces the checkpoint.

Running the CLI requires local artifacts that are intentionally not in
version control (see `ARTIFACTS.md`): `models/yolov8s.pt`
and a YOLO-format dataset under `data/`.

## Engineering Principles

The following principles guide changes to the repository:

1. Keep interfaces and architectural boundaries explicit.
2. Prefer deterministic and reproducible behavior over implicit state.
3. Keep hardware- and runtime-specific concerns isolated.
4. Validate numerical and performance changes with appropriate baselines.
5. Keep the default test suite independent of models and GPU hardware.
6. Treat architecture invariants as part of the project's engineering contract.
7. Every change should be explainable in terms of correctness,
   maintainability, performance, or reproducibility.

## Architecture Invariants

These invariants are deliberate design decisions. Changes that violate
them should include an explicit rationale in the change description or
commit history.

### 1. Thin CLI, implementation in `src/`

`cli/<cmd>.py` is responsible for argument parsing and conversion into the
appropriate dataclass defined in `cli/schema.py`.

Business logic belongs in `src/`.

A new command follows the established pattern:

```text
cli/<cmd>.py
    add_parser(sub)
    run(args)
        ↓
src/<thing>.py
```

The command is then registered in `main.py::COMMANDS`.

### 2. Single source of truth for ONNX Runtime providers

`utils/model_utils.py::select_providers` is the single source of truth for the
ONNX Runtime execution-provider list.

Engine, benchmark, consistency, and quantization code must use this
selection mechanism rather than maintaining independent provider lists.

Adding or changing a CUDA execution-provider option therefore belongs in
`select_providers()`.

### 3. Preserve the YOLOv8 detect-head FP32 boundary

The YOLOv8 detect head is intentionally excluded from INT8 quantization.

`src/quantize.py::HEAD_NAME_PREFIXES = ["/model.22/"]` defines the primary
skip boundary.

The node-resolution helpers must operate on the pre-processed graph,
because `quant_pre_process` may rename graph nodes.

The resolved nodes are passed to `quantize_static()` through
`nodes_to_exclude`.

This boundary exists because quantizing the detect head can significantly
degrade classification confidence in the validated YOLOv8s configuration.

### 4. Keep the three DLPack execution paths distinct

The repository intentionally uses three different DLPack integration
patterns:

- `src/engine.py::_forward`
  - CUDA path
  - `run_with_iobinding`
  - `bind_output("cuda")`
  - explicit `copy_outputs_to_cpu()`
  - deployed inference path

- `src/consistency.py::ort_forward`
  - CUDA path
  - `OrtValue.from_dlpack(...)`
  - standard `session.run(...)`
  - output returned to host

- `src/benchmark.py::_prepare_onnx_input`
  - DLPack `OrtValue`
  - standard `session.run(...)`
  - output returned to host

All three paths avoid an unnecessary host-to-device input copy.

The engine path additionally defers the device-to-host output copy.

These paths should not be collapsed merely for code reuse because they
serve different execution and measurement requirements.

### 5. PT and ONNX benchmark comparisons must use equivalent FP32 paths

PyTorch and ONNX Runtime performance comparisons must remain
FP32-versus-FP32.

The PyTorch backend uses the model directly without autocast:

```python
model.model(...)
```

The benchmark path follows the same rule.

Introducing automatic mixed precision into only the PyTorch path would make
the comparison invalid because the two backends would no longer represent
equivalent numerical workloads.

### 6. Calibration reader length represents batches

`YOLOv8CalibrationDataReader.__len__()` returns the number of calibration
batches rather than the number of individual images.

When `batch_size > 1`, this corresponds to:

```text
ceil(number_of_images / batch_size)
```

This keeps progress reporting aligned with the actual calibration loop.

### 7. Keep numerical comparison utilities dependency-light

`utils/comparison.py` intentionally contains the pure-NumPy tensor
comparison functionality.

This keeps the comparison functionality itself importable without the
`ultralytics` dependency, so the pure-Python test paths stay lightweight.

`src/consistency.py` re-exports the comparison functions for compatibility.

New generic comparison helpers should therefore be added to
`utils/comparison.py`, not implemented inside the consistency workflow.

### 8. Use atomic writes for iterative result generation

Per-batch and per-configuration result files must use
`atomic_json_dump()`.

The expected write pattern is:

```text
temporary file
     ↓
write complete JSON
     ↓
os.replace()
     ↓
final file
```

Do not write directly to the final JSON file when results are generated
incrementally.

This prevents partially written result files from being mistaken for
valid benchmark or consistency results.

### 9. Preserve the validated YOLO output-layout handling

`src/postprocess.py::_ensure_4nc_first` contains a layout heuristic for the
validated YOLOv8s@640 configuration.

The heuristic relies on tensor dimensions and magnitude characteristics.

Synthetic test tensors in `tests/test_postprocess.py` are therefore built
as a channels-first layout with `N` slightly above `4+nc` (e.g.
`N=8`, `4+nc=7`), so the magnitude heuristic resolves unambiguously.

For the validated YOLOv8 configuration, the heuristic is the default path.
For genuinely pruned or structurally different detectors, the explicit
`nc=` parameter should be used and the resulting layout should be
validated.

### 10. Keep lazy public API loading centralized

`src/__init__.py` uses PEP 562 lazy loading.

Public names are registered through the `_LAZY` mapping.

For example:

```python
from src import YOLOv8Engine
```

resolves through `__getattr__` and loads the implementation lazily.

New public names should be registered through `_LAZY`.

In performance-sensitive paths, direct imports such as:

```python
from src.engine import YOLOv8Engine
```

are preferred.

### 11. Keep CLI default paths centralized

Default model and data paths are defined in `cli/__init__.py`:

```text
DEFAULT_MODEL_PT
DEFAULT_MODEL_FP32
DEFAULT_MODEL_INT8
DEFAULT_DATA_DIR
```

These defaults should not be duplicated across individual CLI commands.

If a default path changes, update it in this central location.

### 12. Preserve consistency tolerance semantics

The default consistency tolerances are:

```text
atol = 1e-3
rtol = 1e-2
```

These defaults are intentionally less strict than a pure FP32 numerical
comparison because INT8 quantization introduces expected numerical noise.

The `p99` difference can be affected by low-confidence detections that are
subsequently removed by NMS.

For stricter PT-to-FP32 validation, explicitly request:

```text
--atol 1e-4 --rtol 1e-3
```

The tolerance should therefore be interpreted together with the execution
mode and numerical precision being evaluated.

### 13. Calibration sampling must be deterministic across processes

`src/sampler.py::_stratified_sampling` sorts the per-class candidates
before the seeded shuffle. Set iteration order depends on `PYTHONHASHSEED`,
so the sort is what makes the selection identical in every process —
guarded by
`tests/test_quantize.py::test_stratified_sampling_is_hash_seed_deterministic`
(two subprocesses with different hash seeds must produce the same
selection).

### 14. Docker images reuse the tested requirement files, not hand-pinned sets

Both images (`docker/Dockerfile.toolchain`, `docker/Dockerfile.server`)
install `-r requirements-cpu.txt` verbatim. That overlay pins
`torch==2.5.1+cpu`, `torchvision==0.20.1+cpu`, and `onnxruntime==1.26.0`,
and includes the shared pins of `requirements.txt` (among them
`numpy==1.26.4`), so the images resolve to the exact tested CPU pin set.
The server adds the thin web layer through
`docker/requirements-server.txt` — the only pins beyond the tested CPU
stack are `fastapi`, `uvicorn[standard]`, `python-multipart`, and
`pydantic` (the `[standard]` extra pulls uvicorn's usual companion
packages transitively).

The server carries the full CPU stack on purpose. `ultralytics` is an
unconditional dependency of the engine import path: `src/engine.py`
imports `YOLO` at module level, and `src/postprocess.py` imports
`non_max_suppression` from `ultralytics.utils.nms` at module level, so
`from src import YOLOv8Engine` fails without `ultralytics` regardless of
how NMS is implemented. `requirements.txt` alone deliberately omits
`torch` / `onnxruntime` (see Dependency Boundaries). Because
`ultralytics` is unavoidable on the engine path, the server reuses the
Ultralytics NMS rather than reimplementing it — the rationale is
recorded in the `docker/Dockerfile.server` header.

Any new image must install the project's tested requirement files for
its target runtime verbatim, never a hand-derived pin subset. A
CUDA/DLPack server path must additionally respect the DLPack
synchronization semantics of invariant 4. The multi-stage server
build strips only the compiler (`gcc` / `build-essential`) from the
runtime stage — it is not a stripped runtime.

### 15. The server inference hot path is in-memory

`docker/server.py::_infer_single` calls
`YOLOv8Engine.infer_frames([frame], ...)`, which preprocesses the
already-decoded BGR frame through `preprocess_frames` and reuses
`_run_batch` — the same forward and post-process body as the CLI `infer`
command, so the deployed path and the server path cannot drift.

The hot path operates on the already-decoded in-memory frame end to
end: array decode → `preprocess_frames` → `_run_batch`. A request
never round-trips its upload through disk. (Framework-level upload
buffering that may precede decode — e.g. Starlette spooling large
multipart bodies to a temp file — is outside this invariant.)

`preprocess_imgs` (file-backed) and `preprocess_frames` (array-backed)
share `_preprocess_bgr` + `_assemble_batch` in `src/preprocess.py`:
letterbox lives inside `_preprocess_bgr` and tensor assembly inside
`_assemble_batch`, so a change to either is made once in the shared
core and applies to both entry points — it must not be duplicated per
path.
`preprocess_frames` is registered in `src/__init__.py::_LAZY`.

## Testing and Quality Gates

The default test suite is deliberately designed to run without a model,
GPU, or accelerator-specific runtime.

Expected characteristics:

- no `.pt` or `.onnx` model required;
- no GPU required;
- well under a minute on a typical development machine;
- deterministic unit-level validation wherever practical.

Run the complete test suite with:

```bash
pytest -q
```

Run a specific test file:

```bash
pytest tests/test_postprocess.py -q
```

Run an individual test:

```bash
pytest tests/test_postprocess.py::test_postprocess_clips_to_image_bounds
```

The repository also uses Flake8:

```bash
flake8 .
```

The normal validation gate is:

```bash
pytest -q && flake8 .
```

CI (`.github/workflows/ci.yml`) runs the same gate on Python 3.11 / 3.12
for every push to and pull request against `main`, installing
`requirements-cpu.txt`.

### Test-specific invariants

`tests/conftest.py::data_dir` automatically skips data-dependent tests
when the `data/` directory is unavailable.

`tests/test_postprocess.py` builds synthetic predictions as a channels-first
layout with `N` slightly above `4+nc` (e.g. `N=8`, `4+nc=7`) to avoid
ambiguity in the output-layout heuristic.

`tests/test_consistency.py::test_pure_python_tests_dont_pull_ultralytics`
ensures that the pure-Python test paths do not introduce the heavyweight
`ultralytics` dependency.

Tests that require a model or GPU should remain opt-in rather than being
part of the default test gate.

## Dependency Boundaries

The project intentionally separates lightweight testable components from
heavy runtime dependencies.

In particular:

- pure utility and numerical-comparison code should remain lightweight;
- heavy imports such as `torchvision` and `imagehash` should be delayed
  until the functionality requiring them is executed;
- model- and GPU-dependent functionality should not become a prerequisite
  for the default unit-test suite.

The dependency files distinguish runtime environments:

```text
requirements.txt
requirements-cpu.txt
requirements-kaggle.txt
```

`requirements-cpu.txt` installs CPU `torch` / `onnxruntime` builds for
local development. `requirements-kaggle.txt` targets GPU environments
with a preinstalled CUDA torch build (Kaggle T4) and deliberately does
not install torch or pin numpy — see the comments in that file.
Docker images must consume these files verbatim rather than
re-deriving their own pin sets (invariant 14).

## Configuration and Tooling

`.flake8` defines the project's linting configuration.

`pyproject.toml` contains the Black, isort, and pytest configuration.

The project is currently structured as a CLI and importable Python
library rather than a packaged wheel. Introducing packaging metadata is
acceptable if it provides a clear benefit without weakening the current
development workflow.

## Adding a New CLI Command

A new command should follow this sequence:

1. Add `cli/<cmd>.py` with `add_parser(sub)` and `run(args)`.
2. Add the command's dataclass to `cli/schema.py`.
3. Put implementation logic in the appropriate `src/<thing>.py`.
4. Register new public API names in `src/__init__.py::_LAZY` when needed.
5. Register the command in `main.py::COMMANDS`.
6. Keep heavy optional dependencies out of module-level imports when
   possible.
7. Add the command to the README CLI reference.
8. Add appropriate tests.
9. Run:

```bash
pytest -q && flake8 .
```

## Reproducibility

Changes affecting numerical output, performance, model conversion,
quantization, or runtime behavior should include enough information to
reproduce the observed result.

When reporting benchmark or consistency results, record the relevant:

- model configuration;
- image size;
- batch size;
- execution provider;
- precision;
- software/runtime environment;
- hardware where relevant.

Performance claims should not be based on a single unexplained
measurement.

## AI-Assisted Development

This repository may be developed with AI-assisted coding tools, including
Claude Code.

AI assistance is treated as an implementation and debugging aid rather
than as a substitute for architectural ownership or engineering review.

AI-generated changes are subject to the same requirements as manually
written changes:

- architecture invariants must remain valid;
- interfaces must remain intentional;
- tests must pass;
- linting must pass;
- numerical comparisons must remain meaningful;
- performance measurements must remain comparable;
- changes must be explainable by the author.

The purpose of this document is therefore not to prescribe what an AI
tool should generate, but to make the project's engineering constraints
explicit and enforceable.

## Definition of Done

A change is considered complete when:

1. The implementation satisfies the relevant architecture invariants.
2. Appropriate tests have been added or updated.
3. `pytest -q` passes.
4. `flake8 .` passes.
5. Runtime-dependent behavior has been validated when applicable.
6. Numerical or performance claims have an appropriate baseline.
7. Documentation is updated when the user-facing behavior or CLI changes.
8. The resulting implementation remains understandable and explainable
   by the author.
