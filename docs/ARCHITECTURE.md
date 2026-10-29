# edge-ai-deployment — Architecture Deep Dive

> A walkthrough of one YOLOv8s 12-class edge-deployment pipeline:
> export → ONNX Runtime → INT8 QDQ PTQ → consistency → benchmark.
> Measured on a local Intel Core i5-13420H (CPU) and a Kaggle Tesla T4.
> No embedded board.

---

## Table of Contents

1. [System architecture](#1-system-architecture)
2. [Data-side design](#2-data-side-design)
3. [Model export (PT → ONNX FP32)](#3-model-export-pt--onnx-fp32)
4. [ONNX Runtime inference engine](#4-onnx-runtime-inference-engine)
5. [INT8 static PTQ (QDQ)](#5-int8-static-ptq-qdq)
6. [Calibration sampler design](#6-calibration-sampler-design)
7. [Consistency validation framework](#7-consistency-validation-framework)
8. [Benchmark harness](#8-benchmark-harness)
9. [Threading, memory and logging hygiene](#9-threading-memory-and-logging-hygiene)
10. [Testing strategy](#10-testing-strategy)
11. [Known results (measured numbers)](#11-known-results-measured-numbers)
12. [Operating instructions](#12-operating-instructions)
13. [Glossary](#13-glossary)

---

## 1. System architecture

The CLI layering (thin `cli/` wrappers → typed dataclasses → `src/` machinery) is
overviewed in the [README](../README.md#architecture). This section shows the
tensor-level data flow those layers implement:

```
raw image (HWC BGR)
     │  letterbox — aspect-preserving resize + 114-grey pad      src/preprocess.py
     ▼
(3, imgsz, imgsz) uint8 RGB ── batch, /255 ──▶  (bs, 3, 640, 640) float32
     │  forward: PyTorch │ ORT-CPU (numpy) │ ORT-CUDA (IO Binding + DLPack)
     │                                                          src/engine.py
     ▼
(bs, 4+nc=16, N=8400)  raw output
     │  _ensure_4nc_first → post_process:
     │  NMS (Ultralytics) + scale_boxes + clip                  src/postprocess.py
     ▼
per-image detections [(x1, y1, x2, y2, conf, cls), ...]  in original image coords
```

Key design rules:

* **CLI is thin.** `cli/<cmd>.py` only parses args and forwards a dataclass to `src/`. No business
  logic in CLI.
* **Configs are typed.** Every command except `inspect` (two flat flags, no config object) has a
  `@dataclass` in `cli/schema.py` (`ExportConfig`,
  `QuantizeConfig`, `InferConfig`, `BenchmarkConfig`, `ConsistencyConfig`, the three
  OpenVINO subcommand configs, and the three TensorRT subcommand configs), which provides typing,
  validation, and a printable `repr()` for the log.
* **Logging is centralized.** `utils.logging.setup_logging()` configures the root logger with a
  rotating file + stdout, idempotent under repeat calls; `get_logger(__name__)` everywhere.
* **`src/` is importable from `cli/` and `tests/`.** No top-level scripts doing heavy lifting;
  everything that needs `from src import …` works because `tests/conftest.py` adds the project root
  to `sys.path`.

---

## 2. Data-side design

### 2.1 Letterbox preprocessing

`src/preprocess.py::letterbox()` is the canonical Ultralytics resize-and-pad:

```
   raw image (HWC BGR)
        │
        │  resize so max(H,W) → imgsz, preserve aspect ratio
        ▼
   resized image
        │
        │  pad with (114,114,114) grey to (imgsz, imgsz, 3)
        │  dw/2, dh/2 symmetric padding on each axis
        ▼
   letterboxed image
        │
        │  BGR→RGB, HWC→CHW   (still uint8 at this point)
        ▼
   (3, imgsz, imgsz) uint8
```

Details:

* **`scaleup=True`** by default — the letterbox may enlarge as well as shrink.
* **Half-padding**: `pad_left = int(pad[0] / 2)`. The model was trained on this exact convention;
  the padding at inference must match the training-time convention.
* **Normalization happens once, later.** `preprocess_single` returns uint8 CHW; the `/255` + float32
  conversion happen once per batch, at batch assembly in `preprocess_imgs` (§2.2).

### 2.2 Batched preprocessing with controlled threading

`preprocess_imgs()` uses a `ThreadPoolExecutor` (default `num_workers=4`, clamped to ≤2×CPU by
`utils.threading.clamp_workers`) to apply `preprocess_single()` concurrently, then a single-threaded
`numpy.stack()` + `torch.from_numpy()` + `.to(torch.float32)` + `tensor /= 255.0` for the per-batch
assembly.

Split rationale: OpenCV's letterbox is the expensive part (decoding + INTER_LINEAR +
`copyMakeBorder`) and is per-image independent. The stack/normalize is per-batch and amortized;
it stays single-threaded because it is NumPy/Torch work that does not parallelize under the GIL.

```python
cv2.setNumThreads(0)         # disable OpenCV's internal pool
with ThreadPoolExecutor(...) as ex:
    results = list(ex.map(
        lambda p: preprocess_single(p, imgsz=imgsz_t, original=original),
        img_paths))
```

The same per-image core backs an array-backed entry point: `preprocess_frames()` takes
already-decoded BGR frames (synthetic `frame_{i}` paths, no disk round-trip) through the shared
`_preprocess_bgr` / `_assemble_batch` helpers that `preprocess_imgs` also uses, so a change to
letterbox or tensor assembly is made once in the shared core. `preprocess_frames` is registered in
`src/__init__.py::_LAZY`.

### 2.3 Calibration sampling pipeline

See [Section 6](#6-calibration-sampler-design).

---

## 3. Model export (PT → ONNX FP32)

`cli/export.py` → `src/export.py::model_export(...)`.

```python
exported = model.export(
    format="onnx",
    imgsz=kwargs.get("imgsz", 640),
    opset=kwargs.get("opset", 17),
    dynamic=kwargs.get("dynamic", True),
    simplify=kwargs.get("simplify", True),
    device=device,
    nms=kwargs.get("nms", False)
)
```

**Why each flag:**

| Flag                  | Why                                                                   |
| --------------------- | --------------------------------------------------------------------- |
| `opset=17`            | All ONNX Runtime 1.18+ kernels support opset 17; 13/11 also supported.|
| `dynamic=True`        | One ONNX, many `imgsz` + batch sizes. Static shapes bake batch dim.   |
| `simplify=True`       | onnxslim folds `Reshape->Mul->Reshape` chains and constant folding.   |
| `nms=False`           | Raw logits keep per-class IoU control and detection-level             |
|                       | consistency in post-processing.                                       |

**Validation** (`validate_onnx_model`):

```python
providers = select_providers(device)          # same CUDA EP options as deployment
model = onnx.load(str(path))
onnx.checker.check_model(model)                # structural
try:
    onnx.shape_inference.infer_shapes(model)   # best-effort shape validation
except Exception as e:
    logger.warning("Shape inference skipped: %s", e)   # non-fatal
session = ort.InferenceSession(path, providers=providers)
def _probe(batch):      # random float32 input for every session input, real shapes
    inputs = {inp.name: np.random.randn(*shape).astype(np.float32) ...}
    session.run(None, inputs)
_probe(1)
if isinstance(session.get_inputs()[0].shape[0], str):   # symbolic batch dim
    _probe(3)          # logs "Dynamic batch verified at batch=3"
```

Both `check_model` and a runtime smoke test. Catches: invalid IR, missing initializer,
dynamic-shape-input failure to feed, OP type that no provider implements. A dynamic export
(symbolic batch axis) is additionally probed at batch=3. Shape inference is run for its
validation side-effect; failures are logged and non-fatal, so a model whose shape inference
fails (e.g. a rewritten graph) can still be runtime-validated.

---

## 4. ONNX Runtime inference engine

`src/engine.py::YOLOv8Engine` is the **single entry point** that hides PyTorch vs ORT vs CUDA vs CPU
behind one API.

### 4.1 Provider selection

```python
def select_providers(device: Union[str, int]):
    # device is "cpu", "cuda", "cuda:<id>", or an int GPU id (an int or
    # "cuda:<id>" selects CUDA on that device; "cuda" uses device 0).
    # A malformed "cuda:<id>" raises ValueError.
    # Availability is checked via ort.get_available_providers() (see bullet below).
    if <cuda requested> and "CUDAExecutionProvider" in ort.get_available_providers():
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": cuda_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "1",
                },
            ),
            "CPUExecutionProvider",       # fallback
        ]
    return ["CPUExecutionProvider"]
```

* `kNextPowerOfTwo` — CUDA allocator rounds requests up to power-of-two pool buckets; reduces
  fragmentation, costs at most 2× RAM.
* `HEURISTIC` — selects conv algorithms from a precomputed heuristic table; `EXHAUSTIVE`
  benchmarks every algorithm on first call, which raises startup time.
* `ort.get_available_providers()` — only ORT's own list reflects whether the CUDA EP is actually
  installed/usable; torch can see CUDA while ORT-gpu is missing.
* Always include the CPU provider as a fallback so CUDA-toolkit / driver mismatches downgrade
  gracefully.

### 4.2 CPU path — numpy all the way

```python
inp = batch_tensor.detach().cpu().numpy()               # shared memory
inp = np.ascontiguousarray(inp, dtype=np.float32)       # contiguous on NCHW
return self.session.run(self.output_names, {self.input_name: inp})
```

`detach().cpu().numpy()` is a **zero-copy** view when the tensor is already on CPU.
`ascontiguousarray` guarantees NCHW contiguity for inputs from any preprocess path.

### 4.3 CUDA path — IO Binding via DLPack

```python
io_binding = self.session.io_binding()
batch_tensor = batch_tensor.contiguous()
ort_input = ort.OrtValue.from_dlpack(to_dlpack(batch_tensor))
io_binding.bind_ortvalue_input(self.input_name, ort_input)
for out in self.output_names:
    io_binding.bind_output(out, "cuda")        # copy back via copy_outputs_to_cpu()
self.session.run_with_iobinding(io_binding)
return io_binding.copy_outputs_to_cpu()
```

Details:

1. **DLPack** is the cross-framework tensor protocol; `from_dlpack()` gives ORT the same CUDA
   buffer the torch tensor owns — no copy.
2. `bind_output(..., "cuda")` keeps outputs on the device; the explicit `copy_outputs_to_cpu()` is
   the only host-side transfer for the result.
3. **Warm-up matters.** ORT lazy-loads cuDNN algorithms, allocates the CUDA arena, and (optionally)
   JIT-compiles CUDA graphs on the first call. `_warmup()`
   exercises the **IO Binding + DLPack** path on CUDA (a real CUDA tensor bound via
   `run_with_iobinding`), the same code path the deployed `infer` uses.

> **Where each DLPack entry point lives**: `engine._forward` uses `run_with_iobinding` with
> device-bound outputs (the deployed `infer` path — outputs stay on-device until a single
> `copy_outputs_to_cpu`). `consistency.ort_forward` and `benchmark._prepare_onnx_input` pass the
> same DLPack `OrtValue` into plain `session.run`, whose output returns to host directly. Both are
> zero host→device copy on input; the engine additionally defers the device→host output copy.

---

## 5. INT8 static PTQ (QDQ)

`src/quantize.py::quantize_onnx_to_int8(...)`.

### 5.1 Pipeline

```
   yolov8s_fp32.onnx
        │
        ▼  quant_pre_process (ORT shape inference + baseline graph opt)
   yolov8s_fp32_preprocessed.onnx
        │
        ▼  CalibrationSampler.sample()
   calibration_imgs (Path list)
        │
        ▼  YOLOv8CalibrationDataReader (yields {input: np.float32 batch})
        │
        ▼  quantize_static(
                model_input, model_output,
                calibration_data_reader,
                quant_format=QDQ,
                activation_type=QUInt8,
                weight_type=QInt8,
                per_channel=True,
                calibrate_method=MinMax | Entropy,
                extra_options={ActivationSymmetric: False,
                               WeightSymmetric: True,
                               OpTypesToExcludeOutputQuantization: [...],
                               EnableSubgraph: True})
        ▼
   yolov8s_int8.onnx
```

### 5.2 Why **QDQ**

| Format       | What it looks like                       | Pros                                       | Cons                              |
| ------------ | ---------------------------------------- | ------------------------------------------ | --------------------------------- |
| `QOperator`  | QuantizeLinear / ConvInteger fused       | Lower graph node count for some backends.  | Limited fusion opportunities.     |
| **`QDQ`**    | Each quantizer is `QuantizeLinear`+`DequantizeLinear` pair visible. | Supports conv fusion in the **CUDA EP** and x86 VNNI; INT8 kernels available across providers. | Slightly larger graph. |

This toolkit uses QDQ for its cross-provider INT8 target.

### 5.3 Activation vs Weight quantization

* **Activations are asymmetric** (`QUInt8`, `ActivationSymmetric=False`). ReLU output is
  non-negative; the negative range is wasted; using [0, 255] asymmetric gives the tightest
  discretization.
* **Weights are symmetric** (`QInt8`, `WeightSymmetric=True`). Centered at zero; symmetric is the
  universal default because GPU/CPU GEMM kernels all natively multiply signed×signed.
* **Per-channel weights.** A single conv filter's channels can have widely different dynamic
  ranges; per-channel scales give each channel a scale matched to its own range. Per-channel is
  used for its signed×signed per-channel GEMM kernel coverage.

### 5.4 Calibration methods

| Method     | Cost                        | When to use                                                          |
| ---------- | --------------------------- | -------------------------------------------------------------------- |
| `MinMax`   | Trivial — histogram endpoints. | Fast iteration, uniform-distribution activations.                 |
| `Entropy`  | Slower — KL-divergence minimization across histogram bins. | Skewed / outlier-heavy distributions (common at conv outputs); better final accuracy when calibration cost is acceptable. |

### 5.5 Keep the entire Detect head in FP32

The YOLOv8 detection head (`/model.22/...`) contains the accuracy-critical nodes:

```
   yolov8s head (model.22)
   ├── classification branch: ... → cv3 Conv → Concat → Sigmoid → cls scores
   └── box regression branch: ... → cv2 Conv → DFL (Reshape / Slice / Softmax)
```

Quantizing the **cls-logit branch** (the cv3 convs → `Concat` feeding `Sigmoid`) changes
post-Sigmoid class scores in the validated YOLOv8s configuration. The **primary
protection** is therefore to skip the whole head from quantization by **name-prefix** exclusion:

```python
HEAD_NAME_PREFIXES = ["/model.22/"]   # whole Detect head → FP32 (primary)
# resolved to concrete node names from the pre-processed graph, passed to nodes_to_exclude

OP_TYPES_TO_EXCLUDE_OUTPUT_QUANTIZATION = ["Sigmoid", "Softmax"]
# graph-wide output-quantization exclusion: still protects the head when
# name-based resolution finds nothing (renamed head nodes); in a YOLOv8
# export it also covers the backbone/neck Sigmoid outputs produced by
# SiLU decomposition.
```

The name-prefix exclusion covers the whole head, so the cls-logit path — the cv3 convs feeding
the `Sigmoid` — stays in FP32 end to end. The head accounts for a small share of total FLOPs, so
the FP32 head costs negligible speed.

The head is mostly `Conv/Mul/Concat` — indistinguishable from the backbone/neck by op type, so
op type can't express "skip the head"; the primary exclusion is by name prefix. The
`Sigmoid/Softmax` graph-wide output-quantization exclusion is the second layer: a no-op for the
head once the prefix covers it, and it covers a renamed/merged head node in a future opset.

---

## 6. Calibration sampler design

`src/sampler.py::CalibrationSampler`.

### 6.1 Three-stage pipeline

```
    data.yaml (val dir) + images/val + labels/val
            │
            ▼  Stage 1 — sqrt-frequency stratified sampling
            │
            │  per-class quota  =  max(1, calibration_size * sqrt(f_c) / sum(sqrt(f_·)))
            │  each class contributes at least 1 image
            │
            ▼  Stage 2 — perceptual-hash deduplication
            │
            │  imagehash.phash(img.convert("RGB"))
            │  keep image only if Hamming distance to all kept hashes > 5
            │  drops near-duplicate frames and burst shots
            │
            ▼  Stage 3 — feature-based farthest-first traversal
            │
            │  ResNet-50 (head removed: fc → Identity)
            │  L2-normalized 2048-d features
            │  vectorized greedy: maintain distances[] = min over selections
            │  pick argmax; update distances via elementwise min
            │
            ▼  candidate list of size ≈ calibration_size
```

**Realized counts on the shipped val split (397 images).** `calibration_size` is a *cap*, not a
target. At the default budget of 300 the quota table allocates 295, per-class **availability** trims
the selection to **243** (e.g. `backpack` has 10 val images against a quota of 12), phash dedup
drops **0** (the subset was already phash-deduplicated at construction — [TRAINING.md
§2](TRAINING.md#2-dataset-construction-traindatasetbuild_coco_subsetpy), threshold 6 vs the
sampler's 5), and stage 3 is a **no-op** because the stratified output is under the budget. Raising
the cap to 450 yields 327/397 — the class-availability ceiling is the binding constraint at every
budget ≥ num_classes. The selection is deterministic across processes (seed 42 +
per-class candidates sorted before the seeded shuffle, so `PYTHONHASHSEED` cannot perturb it —
guarded by `test_stratified_sampling_is_hash_seed_deterministic`). PTQ calibration is unsupervised
(activation-range estimation only — no labels, no gradients), so calibrating from the val split
matches the ORT/TensorRT official-example convention; the overlap with the mAP evaluation set is
recorded in [ARTIFACTS.md](../ARTIFACTS.md).

Stage 3 (ResNet-50 farthest-first) only fires when candidates *exceed* `calibration_size` — quotas
sum to ≤ budget by construction, so on the shipped 397-image val split it runs only for degenerate
budgets (< 12 classes) or when `--imgs-input` points at a larger pool. Same dormancy class as the
§6.3 cache.

### 6.2 Why this matters

* **Class coverage under a fixed budget.** sqrt-stratification gives every class ≥1 image and
  keeps rare classes represented within the calibration budget.
* **Duplicate-free budget.** Near-identical images give redundant activation-range estimates;
  phash dedup spends the budget on distinct images.
* **Feature diversity.** Greedy farthest-first on ResNet-50 features maximizes
  feature diversity across the calibration set — the design goal of
  farthest-first selection.

### 6.3 Caching

`CalibrationSampler` JSON-caches the final list under `cache_path` (if provided). On rerun, the
cached paths are validated; the cache is rebuilt if any file is missing. Avoids re-running ResNet-50
forward passes on every quantize run. (The `quantize` CLI path doesn't pass `cache_path` —
caching is opt-in for library callers.)

### 6.4 Performance budgets

| Stage          | Cost (300-image budget)                             |
| -------------- | --------------------------------------------------- |
| phash          | ≈4 s for 243 images (WSL2 i5-13420H, 2026-09; one 32×32 grayscale hash per image) |
| ResNet-50      | not exercised on the shipped val split — stage 3 is a no-op at every budget ≥ num_classes (§6.1); measurable only with a larger `--imgs-input` pool (300 / 32 ≈ 10 batched forwards there) |
| farthest-first | same — O(kN), runs only when candidates exceed the budget |

---

## 7. Consistency validation framework

`src/consistency.py::validate_consistency(...)`.

### 7.1 Two models, two modes

```
   ┌────────────────────┐  tensor  ┌────────────────────┐
   │ model A: PT (.pt)  │─────────▶│ model B: ONNX FP32 │
   └────────────────────┘          └────────────────────┘
       strict tolerance: np.testing.assert_allclose
       (PT<->FP32 runs pass --atol 1e-4 --rtol 1e-3; the CLI default
       1e-3 / 1e-2 is the loose pair for INT8 comparisons)
       catches: export bugs, dtype drift, graph rewrite errors

   ┌────────────────────┐  detection  ┌────────────────────┐
   │ model A: ONNX FP32 │────────────▶│ model B: ONNX INT8 │
   └────────────────────┘             └────────────────────┘
       authoritative: per-image IoU ≥ 0.92, class match ≥ 0.98, score diff ≤ 0.08,
                     count_diff ≤ 3, recall_match ≥ 0.97  ← decides pass/fail
       advisory (informational only): cosine ≥ 0.995, mean_diff ≤ 0.01, p99 ≤ 0.05
       catches: severe INT8 accuracy loss
       NOTE: the raw-output p99 is NOT a hard gate — low-confidence boxes
       that NMS drops can raise it without affecting the final detections,
       so identical detections can show p99 > 0.05.
```

A detection-mode FAIL is an input to the release decision and is read together with the §11
task-level numbers; the measured FP32↔INT8 runs on both harnesses are recorded in §11.
Recalibrating the `score_diff` / recall thresholds against a labeled regression set is future work.

### 7.2 Tensor comparison (`compare_tensors`)

```python
diff = np.abs(out1 - out2)
stats = {
    "max_diff":  float(np.max(diff)),
    "mean_diff": float(np.mean(diff)),
    "p95":       _safe_pct(diff, 95),
    "p99":       _safe_pct(diff, 99),
    "std_diff":  float(np.std(diff)),
    "cosine_similarity": cosine_similarity(out1, out2),
}
```

The class-score channels are reported separately (`cls_max_diff` / `cls_mean_diff` / `cls_p99` /
`cls_cos`): the full-tensor cosine is dominated by the large-magnitude box coordinates and can
mask a class-score divergence small enough to cross a `conf` cliff. With `conf_cliff` set,
`compare_tensors` additionally records the per-side above-cliff counts and directional flips
(`above_cliff_1/2`, `cliff_promoted_1to2`, `cliff_demoted_1to2`) and fails tensor mode when the
per-image flip count exceeds `cliff_count_thresh=5/img` — a gate alongside allclose. The
consistency workflow passes the deployed conf threshold as `conf_cliff`, and the report carries
the dual verdicts `allclose_passed` (raw-tensor parity) / `detection_consistent` (cliff clean) —
a divergence within allclose tolerance that flips boxes across the cliff fails even though
allclose passes, and a reduced-precision pair (FP16/INT8) can fail allclose while detections
stay consistent. See [docs/TENSORRT.md](TENSORRT.md) "Dual verdict" for the full description.

Plus NaN guard, shape-mismatch guard, and an `error_msg` for triage.

### 7.3 Detection comparison (`compare_detections`)

For each image pair:

1. Run `post_process(...)` on both models.
2. Greedy 1:1 matching: walk `dets1`, find the highest-IoU `det2`, consume it (no re-use). Compute
   `mean_iou`, `class_match_rate`, `score_diff_mean`.
3. Aggregate: `recall_match_rate = matched / max(len(dets1), 1)`.
4. Pass if all of `mean_iou ≥ 0.92`, `class_match ≥ 0.98`, `score_diff_mean ≤ 0.08`,
   `count_diff ≤ 3`, `recall_match ≥ 0.97`.

### 7.4 Atomic, recoverable report writing

```python
def atomic_json_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(..., f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
```

And after every (imgsz, batch) combination we re-dump the partial results — mid-run `SIGKILL` keeps
partial findings. The re-dump accumulates *within* one run; across runs the CLI appends a per-run
timestamp to the report stem (`results/consistency_report_<TS>.json`, `<TS>` =
`_YYYYMMDD_HHMMSS_<ms>`), so each run writes its own report file. `--report-path` selects the stem +
directory (e.g. `results/consistency_tensor.json` → `consistency_tensor_<TS>.json`), keeping tensor
and detection runs side by side. Each run's report is a separate
file, and `<TS>` is fixed-width and sortable, so the newest report of a stem is selected by name
sort (`max(glob("<stem>_*.json"))` or `ls ... | sort | tail -1`). Direct library callers that omit
`report_path` still get the unsuffixed fixed path.

When the run completes, a final dump overwrites the last incremental write and adds the
`overall_pass` verdict. The incremental writes omit it because the verdict only exists once every
(imgsz, batch) config has run; a report without an `overall_pass` key means the run ended before
completion. The CLI maps FAIL to exit code 1.

### 7.5 Failed-image triage

When `copy_failed_samples=True` and a batch fails, every image in that batch is copied to
`results/consistency_failed[_<TS>]/<uuid>_<original file name>` — the failed-sample directory
mirrors the report filename (the CLI's timestamped report writes `consistency_failed_<TS>/`; a
library caller on the unsuffixed default gets `consistency_failed/`).

---

## 8. Benchmark harness

`src/benchmark.py::Benchmark`.

### 8.1 What it measures

For each backend:

```
   ┌─ warmup (10 runs of real preprocess + forward + post_process, with real shapes)
   ├─ measured phase (25 runs)
   │     per-run wallclock, including all batches' preprocess + forward
   │     + post_process (NMS)
   │     if CUDA: torch.cuda.synchronize() after warmup + after each run
   │            (both _run_pytorch and _run_onnx — the sync makes perf_counter
   │             measure completed GPU work, including async kernels)
   ├─ percentile latency (p50, p90, p95, p99)
   ├─ total throughput: FPS = num_images / mean_total_s
   └─ peak memory
         CPU  : RSS Δ from baseline (psutil.Process().memory_info().rss)
         CUDA : torch.cuda.max_memory_allocated()
               (peak reset after warmup so warmup allocs are excluded;
                both backends reset, so the number is comparable)
```

And — when `--validation` is passed — a **per-class mAP** from the backend-agnostic native COCO
evaluator (`utils/map_eval.py`), driven over each backend's shared `infer()` API (conf=0.001,
iou=0.7 — the COCO-standard PR-curve config from `--conf-threshold` / `--iou-threshold`). Every
benchmark backend with a Python engine — including raw OpenVINO IRs — goes through the same
evaluator, so mAP50 / mAP50-95 are
directly comparable across backends. CLI runs write the per-class rows to
`results/<backend>_perclass_<TS>.csv` (library callers that omit the timestamp suffix get the
unsuffixed `<backend>_perclass.csv`); the summary CSV records three columns: `mAP50`,
`mAP50-95`, and `mAP_source`.

**mAP provenance (`mAP_source`).** Each summary row records *why* its mAP is present or absent:
`native evaluator` (measured), `unsupported backend (no Python engine to drive)` (a backend without
an `infer()` API — its precision signal is the §7 consistency harness),
`failed: native evaluator crashed (see log)` (engine construction or evaluation raised; the run
continues), `no GT: every class has zero ground truth in the val set`, or
`disabled (--validation off)`. A row without a measurement carries its reason in `mAP_source`.

**Comparability caveat.** The native evaluator computes class-mean mAP with pycocotools' greedy
conf-descending matcher (already-matched GTs are skipped, so a detection can bind a lower-IoU
unmatched GT), but it is not pycocotools: maxDets is the engine's per-image cap (300, vs COCO's
official 100), crowd/ignore regions are not modeled, and confidence ties break deterministically by
(image, box position) — so its numbers can deviate a little from Ultralytics/pycocotools runs on the
same detections. The §11 tables are measured with this evaluator.

### 8.2 Memory hygiene between backends

```python
def _baseline_memory(self):
    gc.collect()
    _try_malloc_trim()                                 # glibc, Linux only
    if self.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    return {"system_rss_mb": psutil.Process().memory_info().rss / 1024**2}
```

`malloc_trim(0)` returns freed heap pages to the OS, keeping `rss_increase_mb` meaningful across
backend runs.

### 8.3 Why percentiles

Mean latency hides the worst frame; for an edge device driving a real camera pipeline, **p99**
characterizes the tail. The harness reports p50/p90/p95/p99 alongside the mean so callers can
compare tail behavior across backends.

---

## 9. Threading, memory and logging hygiene

| Concern                                         | Implementation                                                                                                |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **OpenCV internal thread pool**  | `cv2.setNumThreads(0)` — OpenCV's internal pool is **disabled**; the `ThreadPoolExecutor` in `preprocess_imgs` provides the parallelism. Called once at the start of `preprocess_imgs`. |
| **PyTorch CPU thread pool**                     | `torch.set_num_threads(min(4, ncpu))` — applied by the benchmark's `_run_pytorch` so the PT-vs-ORT comparison runs under the same single-stream thread budget. |
| **ORT intra/inter op threads**                  | `sess_options.intra_op_num_threads = min(4, ncpu)`, `inter_op_num_threads = 2` (1 when intra ≤ 1) — applied as the `YOLOv8Engine` default and by the benchmark's `SessionOptions`. The validated single-stream setting is ≤4/2. |
| **Calibration / inference pool oversubscription** | `cv2.setNumThreads(0)` is enforced at the top of `preprocess_imgs` (OpenCV internal pool disabled — the `ThreadPoolExecutor` is the only parallel path). `YOLOv8Engine` accepts `intra_op_threads` / `inter_op_threads` constructor args to override the 4/2 default; the benchmark bakes the same values into its `SessionOptions`. |
| **Rotating-file logging**                       | `RotatingFileHandler(filename, maxBytes=10 MiB, backupCount=5)` + a stdout handler with shared formatter.     |
| **Idempotent logger init**                      | `_logging_initialized` flag — re-calls return early; `force=True` re-installs handlers (available for post-init reconfiguration). |
| **Atomic JSON**                                 | `tmp + os.replace`, plus a re-dump after every (imgsz, batch-size) combination so partial results survive crashes. |
| **psutil + gc** between runs                    | RSS baselines are clean; CUDA peak is reset per run.                                                          |

---

## 10. Testing strategy

Three principles:

1. **Tests must run on a laptop, without a model or GPU.** Heavier feature modules
   (`src/consistency.py`, `src/quantize.py`) are imported inside test functions, and
   `test_pure_python_tests_dont_pull_ultralytics` enforces that the four pure-Python test modules
   (`test_consistency`, `test_quantize`, `test_utils`, `test_map_eval`) don't pull `ultralytics`
   at import time.
   Several test modules bind heavy dependencies at module level: `test_postprocess` (because
   `src.postprocess` needs Ultralytics NMS), `test_benchmark_map_plumbing` (because
   `src.benchmark` imports torch / onnxruntime / pandas / psutil / ultralytics), and
   `test_consistency_ort_forward` (module-level `import torch`).
   `test_postprocess.py` builds a synthetic `(bs=1, 4+nc=7, N≥8)` prediction and exercises the full
   path.
2. **No mocking.** The functions are written to be unit-testable without mocks — `post_process`
   accepts `torch.Tensor`, `np.ndarray`, or `list`, and the layout heuristic is sized for
   `num_boxes > 32` so it never misfires on a small test tensor.
3. **One concept per test.** `test_compute_iou_*`, `test_compare_tensors_*`,
   `test_compare_detections_*` — every test exercises a single property. The tests double as
   documentation of edge cases (empty matching, identical inputs, NaN inputs, shape mismatch, count
   penalty).

---

## 11. Known results (measured numbers)

> **CPU rows measured 2026-10-15** (WSL2 Ubuntu on the i5-13420H laptop; Python 3.12.3, torch
> 2.6.0+cpu — the FP32 export and INT8 model were produced under it — onnxruntime 1.26.0,
> ultralytics 8.4.114): speed loop = 14 sampler-selected val images (budget 16), batch 1, warmup 10,
> runs 25; latency = per-run wallclock over the speed set (`results/benchmark_summary_<TS>.csv`);
> mAP = full 397-image val split; calibration = 243/397 val images (budget 300, seed 42).
> **CUDA rows measured 2026-10-16** on Kaggle Tesla T4 (run `20261016_030921_424`;
> Python 3.12.13, torch 2.10.0+cu128 — the image's preinstalled CUDA build, recorded in
> `results_on_kaggle/torch_env.json`; onnxruntime-gpu 1.26.0 — pinned by
> `requirements-kaggle.txt`; cuDNN 91002), same speed-loop protocol. The
> FP32 export and the INT8 quantization for the CUDA harness are done **on the Kaggle box with
> `--device cuda`** (CUDA-EP calibration), so the CPU and CUDA FP32/INT8 artifacts differ by
> sha256 and the two harnesses are not a same-artifact comparison. See the CUDA-harness notes
> below the table.

| Backend         | mean_latency (ms) | p95 (ms) | FPS  | RSS Δ (MB) | mAP50 (full) | mAP50-95 |
| --------------- | ----------------- | -------- | ---- | ---------- | ------------ | -------- |
| PyTorch CPU     | 3941.2            | 5467.2   | 3.55 | 140.5      | 0.590        | 0.402    |
| ONNX FP32 CPU   | 2957.1            | 3276.5   | 4.73 | 226.0      | 0.590        | 0.402    |
| ONNX INT8 CPU   | 1801.4            | 1976.8   | 7.77 | 104.9      | 0.589        | 0.399    |
| PyTorch (CUDA)  | 244.7             | 270.1    | 57.2 | 543.9      | 0.590        | 0.402    |
| ONNX FP32 CUDA  | 717.6             | 917.4    | 19.51 | 326.6     | 0.590³       | 0.402³   |
| ONNX INT8 CUDA  | 320.3             | 333.2    | 43.71 | 49.5      | 0.589³       | 0.396³   |

**CUDA-harness notes** (ORT-gpu 1.26.0; 2026-10-16):
1. **FP32-CUDA vs PyTorch CUDA** (19.51 vs 57.2 FPS): the Detect-head subgraph (`/model.22/`
   Gather/Concat/Unsqueeze/Mul chains — ops the CUDA EP implements) is assigned to the CPU EP by
   ORT's cost heuristic, adding per-inference host↔device round-trips of the `(bs, 16, 8400)` head
   tensors. The FP32/INT8 CUDA latency rows correspond to this ORT-gpu build (1.26.0); the
   PyTorch CUDA reference is unaffected.
2. **INT8-on-CUDA detection consistency.** The FP32↔INT8 detection comparison runs through
   `src/consistency.py::ort_forward`, whose DLPack input is produced on the torch stream and
   synchronized before `session.run` reads the device buffer — the same barrier the engine path
   applies (note 3). 13/16 images pass (overall FAIL) — 2 recall-gated failures
   (recall 0.750 / 0.333, count_diff 1) and 1 score-diff-gated failure (0.177, IoU 0.929),
   class match 1.0 on all 16, advisory cosine 0.9986. The INT8 session routes each conv-bias
   `DequantizeLinear` through a `MemcpyFromHost` node (the CUDA EP has no kernel
   for 1-D/per-channel bias DQ in this build) — the QDQ-on-CUDA structural cost.
3. **Cross-stream barrier for device-buffer inputs.** On the CUDA path the engine binds a torch-CUDA
   buffer as the IO-binding input. The buffer is produced on the torch stream
   (`preprocess.py::_assemble_batch`: pinned H2D copy + in-place `/= 255`), while ORT executes on
   its own stream; `src/engine.py::_forward` and `_warmup` therefore call
   `torch.cuda.current_stream().synchronize()` before `run_with_iobinding`. The benchmark speed
   loop uses `session.run`, so the latency / FPS / RSS columns are unaffected; the ONNX-CUDA mAP
   columns (³) measure 0.590 / 0.402 (FP32) and 0.589 / 0.396 (INT8).

mAP columns are per-class means (`results/<backend>_perclass_<TS>.csv` on CLI runs). The PT
and ONNX FP32 rows agree per class (0.590 / 0.402), and PT↔FP32 passes the strict tensor gate, so
the FP32 export reproduces the PT model. The values sit below the training-side record
(0.601 / 0.411, [TRAINING.md §6](TRAINING.md#6-results)); the offset is consistent with the
torch-2.6.0+cpu runtime numerics and the evaluator methodology difference (§8.1 — the
training-side record used the Ultralytics evaluator).

**PT ↔ ONNX FP32** (tensor mode, strict `--atol 1e-4 --rtol 1e-3`) passes on both
harnesses (imgsz 640, batch 1, 16 images): CPU (2026-10-15) — max_diff mean 0.002359,
cosine 1.000 (`results/consistency_tensor.json`); CUDA (2026-10-16) —
max_diff 0.002483, cosine 1.000
(`results_on_kaggle/consistency_tensor.json`).

**FP32 ↔ INT8** (detection mode, CPU harness, 2026-10-15; 16 sampler-selected images, budget 20): **FAIL
at the per-image gates** — 13/16 images pass (81%); all three failures are recall-gated (recall
0.33–0.75, count_diff 1, IoU ≥ 0.974, class match 1.0). Aggregate: mean IoU 0.980, mean score diff 0.026, advisory
cosine 0.9986; full-val mAP50-95 drops 0.0029 (0.402 → 0.399). On the CUDA harness
(2026-10-16 run) 13/16 images pass, overall FAIL — 2 recall-gated failures (recall 0.750 / 0.333,
count_diff 1, class match 1.0) and 1 score-diff-gated failure (0.177); advisory cosine 0.9986
(note 2 above).

> The exact numbers depend on CUDA / cuDNN / ORT versions and dataset. Measured INT8-vs-FP32 on the
> CPU EP: **1.64×** faster, mAP50-95 drop **0.0029**. On the CUDA harness INT8-vs-FP32 is **2.24×**
> faster on an FP32 denominator affected by the CPU-EP fallback (note 1), with an mAP50-95
> drop of 0.0054 (0.402 → 0.396). Detection-mode consistency is judged by the §7 gates; gate
> recalibration is future work (§7.1).

---

## 12. Operating instructions

See the [README Quick Start](../README.md#quick-start) for the full command sequence
(export → inspect → consistency → quantize → infer → benchmark) and the CLI reference;
run `./clean.sh` to reset generated artifacts.

---

## 13. Glossary

| Term                     | One-liner                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| **ONNX**                 | Open Neural Network Exchange — a graph IR + operator set.                                   |
| **ONNX Runtime**         | Microsoft's cross-platform inference engine for ONNX graphs.                                |
| **EP (Execution Provider)**| An ORT backend implementation (CPU, CUDA, TensorRT, OpenVINO, …).                         |
| **IO Binding**           | Wire pre-allocated device buffers into ORT without host copies.                             |
| **DLPack**               | Cross-framework tensor protocol; `from_dlpack(t)` shares memory.                            |
| **PTQ**                  | Post-Training Quantization — calibrate on a small dataset, no backprop.                     |
| **QDQ**                  | Quantize-Dequantize pairs visible in the graph. Preferred ORT INT8 format.                  |
| **QOperator**            | Legacy fused-quant format. Smaller graphs, fewer fusion options.                            |
| **MinMax / Entropy**     | Calibration methods: histogram endpoints vs KL-divergence minimization.                     |
| **Per-channel**          | Each output channel gets its own scale/zero-point. Preserves accuracy on asymmetric weights.|
| **QInt8 / QUInt8**       | Signed (weights) vs unsigned (activations) 8-bit quantization types.                        |
| **CUDA EP options**      | `arena_extend_strategy`, `cudnn_conv_algo_search`, `cudnn_conv_use_max_workspace`.          |
| **Ultralytics NMS**      | `ultralytics.utils.nms.non_max_suppression` — class-aware, vectorized.                      |
| **letterbox**            | Aspect-preserving resize + symmetric pad to (imgsz, imgsz, 3).                              |
| **RSS**                  | Resident Set Size — process's pages resident in physical RAM.                               |
| **p50/p95/p99**          | Latency percentiles; tail latency metrics used in production.                               |
