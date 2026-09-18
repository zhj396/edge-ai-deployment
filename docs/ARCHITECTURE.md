# edge-ai-deployment — Architecture Deep Dive

> A line-by-line walkthrough of one YOLOv8s 12-class edge-deployment pipeline: export → ONNX Runtime → INT8 QDQ PTQ → consistency → benchmark.
> Built on local Intel Core i5-13420H + Kaggle Tesla T4 ×2. No embedded board.

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
11. [Known results (representative numbers)](#11-known-results-representative-numbers)
12. [Lessons learned](#12-lessons-learned)
13. [Operating instructions](#13-operating-instructions)
14. [Glossary](#14-glossary)

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

* **CLI is thin.** `cli/<cmd>.py` only parses args and forwards a dataclass to `src/`. No business logic in CLI.
* **Configs are typed.** Every command has a `@dataclass` in `cli/schema.py` (`ExportConfig`, `QuantizeConfig`, `InferConfig`, `BenchmarkConfig`, `ConsistencyConfig`). Cheap IDE autocomplete, cheap validation, free `repr()` for the log.
* **Logging is centralized.** `utils.logging.setup_logging()` configures the root logger with a rotating file + stdout, idempotent under repeat calls; `get_logger(__name__)` everywhere.
* **`src/` is importable from `cli/` and `tests/`.** No top-level scripts doing heavy lifting; everything that needs `from src import …` works because `tests/conftest.py` adds the project root to `sys.path`.

---

## 2. Data-side design

### 2.1 Letterbox preprocessing

`src/preprocess.py::letterbox()` is the canonical Ultralytics resize-and-pad:

```
   raw image (HWC BGR)
        │
        │  resize so min(H,W) → imgsz, preserve aspect ratio
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

Three non-obvious things:

* **`scaleup=True`** by default — the letterbox may enlarge, not just shrink.
* **Half-padding**: `pad_left = int(pad[0] / 2)`. The model was trained on this exact convention; mismatching it silently degrades mAP.
* **Normalization happens once, later.** `preprocess_single` returns uint8 CHW; the `/255` + float32 conversion happen at batch assembly in `preprocess_imgs` (§2.2) — not per image.

### 2.2 Batched preprocessing with controlled threading

`preprocess_imgs()` uses a `ThreadPoolExecutor` (default `num_workers=4`, clamped to ≤2×CPU by `utils.threading.clamp_workers`) to apply `preprocess_single()` concurrently, then a single-threaded `numpy.stack()` + `torch.from_numpy()` + `.to(torch.float32)` + `tensor /= 255.0` for the per-batch assembly.

Why this split? OpenCV's letterbox is the expensive part (decoding + INTER_LINEAR + `copyMakeBorder`) and is per-image independent. The stack/normalize is per-batch and amortized — running it in parallel would just thrash the GIL.

```python
cv2.setNumThreads(0)         # disable OpenCV's internal pool
with ThreadPoolExecutor(...) as ex:
    results = list(ex.map(
        lambda p: preprocess_single(p, imgsz=imgsz_t, original=original),
        img_paths))
```

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
    quantize=kwargs.get("quantize", "fp32"),
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
| `quantize="fp32"`     | Export precision: "int8" / "fp16" / "fp32".                            |
| `nms=False`           | We want raw logits for post-processing control. The YOLO-internal NMS |
|                       | loses you per-class IoU control and detection-level consistency.      |

**Validation** (`validate_onnx_model`):

```python
model = onnx.load(str(path))
onnx.checker.check_model(model)                # structural
try:
    onnx.shape_inference.infer_shapes(model)   # best-effort shape validation
except Exception as e:
    logger.warning("Shape inference skipped: %s", e)   # non-fatal
session = ort.InferenceSession(path, providers=providers)
session.run(None, {inp.name: np.random.randn(*shape).astype(np.float32)})
```

Both `check_model` and a runtime smoke test. Catches: invalid IR, missing initializer, dynamic-shape-input failure to feed, OP type that no provider implements. Shape inference is run for its validation side-effect and made non-fatal so models whose inference fails (e.g. some rewritten graphs) can still be runtime-validated.

---

## 4. ONNX Runtime inference engine

`src/engine.py::YOLOv8Engine` is the **single entry point** that hides PyTorch vs ORT vs CUDA vs CPU behind one API.

### 4.1 Provider selection

```python
def select_providers(device: str):
    # Availability is checked via ort.get_available_providers(), NOT torch.cuda.is_available(): ORT-gpu may be absent even when torch sees CUDA, and only ORT's own list tells us the CUDA EP is actually usable.
    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "1",
                },
            ),
            "CPUExecutionProvider",       # fallback
        ]
    return ["CPUExecutionProvider"]
```

* `kNextPowerOfTwo` — CUDA allocator rounds requests up to power-of-two pool buckets; reduces fragmentation, costs at most 2× RAM.
* `HEURISTIC` — faster startup (`EXHAUSTIVE` measures every algorithm the first time, better steady-state, slow first inference).
* `ort.get_available_providers()` (not `torch.cuda.is_available()`) — only ORT's own list reflects whether the CUDA EP is actually installed/usable; torch can see CUDA while ORT-gpu is missing.
* Always include the CPU provider as a fallback so CUDA-toolkit / driver mismatches downgrade gracefully.

### 4.2 CPU path — numpy all the way

```python
inp = batch_tensor.detach().cpu().numpy()               # shared memory
inp = np.ascontiguousarray(inp, dtype=np.float32)       # contiguous on NCHW
return self.session.run(self.output_names, {self.input_name: inp})
```

`detach().cpu().numpy()` is a **zero-copy** view when the tensor is already on CPU. `ascontiguousarray` is needed only because some preprocess code paths may have produced non-contiguous intermediates.

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

Three things to remember:

1. **DLPack** is the cross-framework tensor protocol; `from_dlpack()` gives OR-T the same CUDA buffer the torch tensor owns — no copy.
2. `bind_output(..., "cuda")` keeps outputs on the device; the explicit `copy_outputs_to_cpu()` is the only host-side transfer for the result.
3. **Warm-up matters.** ORT lazy-loads cuDNN algorithms, allocates the CUDA arena, and (optionally) JIT-compiles CUDA graphs on the first call. A torch-only warm-up is meaningless — `_warmup()` exercises the **IO Binding + DLPack** path on CUDA (a real CUDA tensor bound via `run_with_iobinding`), not just `session.run` with host numpy.

> **Where each DLPack entry point lives**: `engine._forward` uses `run_with_iobinding` with device-bound outputs (the deployed `infer` path — outputs stay on-device until a single `copy_outputs_to_cpu`). `consistency.ort_forward` and `benchmark._prepare_onnx_input` pass the same DLPack `OrtValue` into plain `session.run`, whose output returns to host directly. Both are zero host→device copy on input; the engine additionally defers the device→host copy.

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

### 5.2 Why **QDQ** (not QOperator)

| Format       | What it looks like                       | Pros                                       | Cons                              |
| ------------ | ---------------------------------------- | ------------------------------------------ | --------------------------------- |
| `QOperator`  | QuantizeLinear / ConvInteger fused       | Lower graph node count for some backends.  | Limited fusion opportunities.     |
| **`QDQ`**    | Each quantizer is `QuantizeLinear`+`DequantizeLinear` pair visible. | Maximum fusion in **CUDA EP** & x86 VNNI; best kernel-coverage across providers. | Slightly larger graph. |

For cross-provider INT8 — the scenario this toolkit targets — QDQ wins.

### 5.3 Activation vs Weight quantization

* **Activations are asymmetric** (`QUInt8`, `ActivationSymmetric=False`). ReLU output is non-negative; the negative range is wasted; using [0, 255] asymmetric gives the tightest discretization.
* **Weights are symmetric** (`QInt8`, `WeightSymmetric=True`). Centered at zero; symmetric is the universal default because GPU/CPU GEMM kernels all natively multiply signed×signed.
* **Per-channel weights.** A single conv filter's channels can have wildly different dynamic ranges; per-channel preserves accuracy for "tiny" channels that would otherwise be drowned. Per-tensor would drown those channels and cost measurable mAP on YOLOv8s.

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

Per-channel INT8 QDQ on the **cls-logit branch** (the cv3 convs → `Concat` feeding `Sigmoid`) collapses the post-Sigmoid class scores to **all-zero**: MinMax calibration of the cls logits is distorted by negative outliers, the DequantizeLinear (asymmetric uint8, `zp≈246`) maps real `~+1.7` logits into a range whose sigmoid collapses to 0, so every image returns 0 detections. The box-decoder branch survives quantization; the cls branch does not. So the **primary protection** is to skip the whole head from quantization by **name-prefix** exclusion:

```python
HEAD_NAME_PREFIXES = ["/model.22/"]   # whole Detect head → FP32 (primary)
# resolved to ~160 node names from the pre-processed graph, passed to nodes_to_exclude

OP_TYPES_TO_EXCLUDE_OUTPUT_QUANTIZATION = ["Sigmoid", "Softmax"]  # residual fallback
NODE_TYPES_TO_EXCLUDE = ["Sigmoid", "Softmax"]                     # residual fallback
```

Why the whole head, not just the `Sigmoid/Softmax` nodes? Excluding only the `Sigmoid/Softmax` *nodes themselves* leaves their upstream cls conv QDQ-wrapped — which is exactly the failure above. Excluding the whole subgraph keeps the cls-logit path in FP32 end to end. The head is <5% of FLOPs, so the FP32 head costs negligible speed. Verification on the shipped checkpoint (CPU harness, 2026-09): on a 32-image val sample the INT8-vs-FP32 top-5 cls scores (post-sigmoid) differ by **max 0.046, mean 0.006** — no collapse (ranges: FP32 0.502–0.715, INT8 0.502–0.713); on the 91-image consistency sample FP32↔INT8 matched 187 detection pairs with mean IoU 0.959 and mean score diff 0.032 (§11).

Why name-prefix and not pure op-type? The head is mostly `Conv/Mul/Concat` — indistinguishable from backbone/neck by op type, so op-type can't express "skip the head". The `Sigmoid/Softmax` op-type exclusion is kept as residual belt-and-suspenders inside the head (a no-op once the prefix covers it, but cheap insurance against a renamed/merged head node in a future opset).

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

**Realized counts on the shipped val split (397 images).** `calibration_size` is a *cap*, not a target. At the default budget of 300 the quota table allocates 295, per-class **availability** trims the selection to **243** (e.g. `backpack` has 10 val images against a quota of 12), phash dedup drops **0** (the subset was already phash-deduplicated at construction — [TRAINING.md §2](TRAINING.md#2-dataset-construction-traindatasetbuild_coco_subsetpy), threshold 6 vs the sampler's 5), and stage 3 is a **no-op** because the stratified output is under the budget. Raising the cap to 450 yields 327/397 — the class-availability ceiling, not the budget, is the binding constraint at every budget ≥ num_classes. The selection is deterministic across processes (seed 42 + per-class candidates sorted before the seeded shuffle, so `PYTHONHASHSEED` cannot perturb it — guarded by `test_stratified_sampling_is_hash_seed_deterministic`). Calibrating from the val split is deliberate: PTQ calibration is unsupervised (activation-range estimation only — no labels, no gradients), matching the ORT/TensorRT official-example convention; the overlap with the mAP evaluation set is disclosed here and in [ARTIFACTS.md](../ARTIFACTS.md) rather than engineered away with a disjoint pool.

Stage 3 (ResNet-50 farthest-first) only fires when candidates *exceed* `calibration_size` — quotas sum to ≤ budget by construction, so on the shipped 397-image val split it runs only for degenerate budgets (< 12 classes) or when `--imgs-input` points at a larger pool. Same dormancy class as the §6.3 cache.

### 6.2 Why this matters

* **Random sampling drowns rare classes.** Person vs. toothbrush in COCO; if you draw 300 i.i.d. images from a long-tailed distribution you get almost no toothbrush. With sqrt-stratification every class gets ≥1 image and rare classes are still represented.
* **Visual duplicates waste budget.** Ten near-identical beach photos give ten times the activation-range estimate for "beach". Phash dedup caps this.
* **I.i.d. ≠ representative.** Greedy farthest-first on ResNet-50 features maximizes feature diversity, which empirically correlates with calibration accuracy.

### 6.3 Caching

`CalibrationSampler` JSON-caches the final list under `cache_path` (if provided). On rerun, the cached paths are validated; the cache is rebuilt if any file is missing. Avoids re-running ResNet-50 forward passes on every quantize run. (The `quantize` CLI path currently doesn't pass `cache_path` — caching is opt-in for library callers.)

### 6.4 Performance budgets

| Stage          | Cost (300-image budget)                             |
| -------------- | --------------------------------------------------- |
| phash          | ≈4 s for 243 images on the WSL2 i5-13420H harness (one 32×32 grayscale hash per image) |
| ResNet-50      | not exercised on the shipped val split — stage 3 is a no-op at every budget ≥ num_classes (§6.1); measurable only with a larger `--imgs-input` pool (300 / 32 ≈ 10 batched forwards there) |
| farthest-first | same — O(kN), runs only when candidates exceed the budget |

*(Timings from the CPU harness, 2026-09 — §11; other environments differ.)*

---

## 7. Consistency validation framework

`src/consistency.py::validate_consistency(...)`.

### 7.1 Two models, two modes

```
   ┌────────────────────┐  tensor  ┌────────────────────┐
   │ model A: PT (.pt)  │─────────▶│ model B: ONNX FP32 │
   └────────────────────┘          └────────────────────┘
       strict tolerance: np.testing.assert_allclose (lib default atol=1e-4, rtol=1e-3;
       CLI default is looser: 1e-3 / 1e-2 — pass --atol/--rtol explicitly
       for PT-vs-FP32)
       catches: export bugs, dtype drift, graph rewrite errors

   ┌────────────────────┐  detection  ┌────────────────────┐
   │ model A: ONNX FP32 │────────────▶│ model B: ONNX INT8 │
   └────────────────────┘             └────────────────────┘
       authoritative: per-image IoU ≥ 0.92, class match ≥ 0.98, score diff ≤ 0.08,
                     count_diff ≤ 3, recall_match ≥ 0.97  ← decides pass/fail
       advisory (informational only): cosine ≥ 0.995, mean_diff ≤ 0.01, p99 ≤ 0.05
       catches: catastrophic INT8 accuracy loss
       NOTE: the raw-output p99 is NOT a hard gate — INT8 routinely raises it
       via noise in low-confidence boxes that NMS drops, so identical final
       detections can show p99 > 0.05. The detection gates catch real loss.
```

**Measured (CPU harness, 2026-09; §11):** on the 91-image FP32↔INT8 run the per-image gates fail 31% of images — recall < 0.97 (18), score diff > 0.08 (9), IoU < 0.92 (8), class match (3) — while full-val mAP50-95 drops 0.0026 and matched pairs average IoU 0.959. INT8 shifts confidence scores and swaps borderline boxes at conf 0.25 more than the gates assume. The gates are deliberately conservative: a detection-mode FAIL should be read together with the §11 task-level numbers, not as a release blocker by itself. Recalibrating the `score_diff` / recall thresholds against a labeled regression set is future work.

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

Plus NaN guard, shape-mismatch guard, and an `error_msg` for triage.

### 7.3 Detection comparison (`compare_detections`)

For each image pair:

1. Run `post_process(...)` on both models.
2. Greedy 1:1 matching: walk `dets1`, find the highest-IoU `det2`, consume it (no re-use). Compute `mean_iou`, `class_match_rate`, `score_diff_mean`.
3. Aggregate: `recall_match_rate = matched / max(len(dets1), 1)`.
4. Pass if all of `mean_iou ≥ 0.92`, `class_match ≥ 0.98`, `score_diff_mean ≤ 0.08`, `count_diff ≤ 3`, `recall_match ≥ 0.97`.

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

And after every (imgsz, batch) combination we re-dump the partial results — mid-run `SIGKILL` keeps partial findings. The re-dump accumulates *within* one run; across runs the report is replaced, so `--report-path` exists to give tensor and detection runs separate files (the shared default `results/consistency_report.json` keeps only the latest run).

### 7.5 Failed-image triage

When `copy_failed_samples=True` and a batch fails, every image in that batch is copied to `results/consistency_failed/<uuid>_name.jpg`. This is the single biggest productivity boost: you eyeball the failure mode without searching the dataset.

---

## 8. Benchmark harness

`src/benchmark.py::Benchmark`.

### 8.1 What it measures

For each backend:

```
   ┌─ warmup (10 runs of real preprocess + forward, with real shapes)
   ├─ measured phase (25 runs)
   │     per-run wallclock, including all batches' preprocess + forward
   │     + post_process (NMS)
   │     if CUDA: torch.cuda.synchronize() after warmup + after each run
   │            (both _run_pytorch and _run_onnx — without it perf_counter
   │             captures kernel-launch overhead and undercounts async GPU work)
   ├─ percentile latency (p50, p90, p95, p99)
   ├─ total throughput: FPS = num_images / mean_total_s
   └─ peak memory
         CPU  : RSS Δ from baseline (psutil.Process().memory_info().rss)
         CUDA : torch.cuda.max_memory_allocated()
               (peak reset after warmup so warmup allocs are excluded;
                both backends reset, so the number is comparable)
```

And — when `--validation` is passed — a **per-class mAP** via Ultralytics' `model.val(...)`, which we read into a CSV per backend (`results/<backend>_perclass.csv`).

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

`malloc_trim(0)` returns freed heap pages to the OS. Without this, glibc's allocator hoards and the RSS keeps climbing across runs — the `rss_increase_mb` numbers would be wrong.

### 8.3 Why percentiles

Mean latency hides the worst frame. For an edge device driving a real camera pipeline, **p99** is what matters. The harness reports p50/p90/p95/p99 alongside the mean so callers can compare tail behavior across backends.

---

## 9. Threading, memory and logging hygiene

| Concern                                         | Implementation                                                                                                |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **OpenCV internal thread pool oversubscribed**  | `cv2.setNumThreads(0)` — OpenCV's own pool is **disabled**; the `ThreadPoolExecutor` in `preprocess_imgs` provides the parallelism, and per-worker cv2 threads would oversubscribe. Called once at the start of `preprocess_imgs`. |
| **PyTorch CPU thread pool**                     | `torch.set_num_threads(min(4, ncpu))` — applied by the benchmark's `_run_pytorch` so the PT-vs-ORT comparison is apples-to-apples. |
| **ORT intra/inter op threads**                  | `sess_options.intra_op_num_threads = min(4, ncpu)`, `inter_op_num_threads = 2` (1 when intra ≤ 1) — applied as the `YOLOv8Engine` default and by the benchmark's `SessionOptions`. ORT's "all cores" default oversubscribes a single-stream workload; ≤4/2 is the sweet spot. |
| **Calibration / inference pool oversubscription** | `cv2.setNumThreads(0)` is enforced at the top of `preprocess_imgs` (OpenCV internal pool disabled — the `ThreadPoolExecutor` is the only parallel path). `YOLOv8Engine` accepts `intra_op_threads` / `inter_op_threads` constructor args to override the 4/2 default; the benchmark bakes the same values into its `SessionOptions`. |
| **Rotating-file logging**                       | `RotatingFileHandler(filename, maxBytes=10 MiB, backupCount=5)` + a stdout handler with shared formatter.     |
| **Idempotent logger init**                      | `_logging_initialized` flag — re-calls return early; `force=True` re-installs (used by `--log-level`).        |
| **Atomic JSON**                                 | `tmp + os.replace`, plus a re-dump after every (imgsz, batch-size) combination so partial results survive crashes. |
| **psutil + gc** between runs                    | RSS baselines are clean; CUDA peak is reset per run.                                                          |

---

## 10. Testing strategy

Three principles:

1. **Tests must run on a laptop, without a model or GPU.** Module-level test imports stay on the dependency-light helpers — `utils/comparison.py` (pure NumPy), `src/postprocess.py` (pure-tensor logic; pulls Ultralytics NMS at import), `utils/` (no model); heavier feature modules (`src/consistency.py`, `src/quantize.py`) are imported inside test functions, and `test_pure_python_tests_dont_pull_ultralytics` enforces that the three pure-Python test modules (`test_consistency`, `test_quantize`, `test_utils`) don't pull `ultralytics` at import time — `test_postprocess` is exempt because `src/postprocess` needs Ultralytics NMS. `test_postprocess.py` builds a synthetic `(bs=1, 4+nc=7, N≥8)` prediction and exercises the full path.
2. **No mocking.** The functions are written to be unit-testable *naturally* — `post_process` accepts `torch.Tensor`, `np.ndarray`, or `list`, and the layout heuristic is sized for `num_boxes > 32` so it never misfires on a small test tensor.
3. **One concept per test.** `test_compute_iou_*`, `test_compare_tensors_*`, `test_compare_detections_*` — every test exercises a single property. The tests double as documentation of edge cases (empty matching, identical inputs, NaN inputs, shape mismatch, count penalty).

---

## 11. Known results (representative numbers)

> **Measured on the CPU harness** (WSL2 Ubuntu on the i5-13420H laptop; Python 3.12.3, torch 2.5.1+cpu, onnxruntime 1.26.0, ultralytics 8.4.114; 2026-09): speed loop = 14 sampler-selected val images (budget 16), batch 1, warmup 10, runs 25; latency = per-run wallclock over the speed set (`results/benchmark_summary.csv`); mAP = full 397-image val split; calibration = 243/397 val images (budget 300, seed 42). **CUDA rows measured on Kaggle Tesla T4 ×2** (onnxruntime-gpu 1.26.0, 2026-09; same speed-loop protocol; the CUDA-harness INT8 was re-exported and quantized there with `--device cuda` — CUDA-EP calibration — so the two INT8 artifacts differ by sha256; both show the normal CPU-pipeline profile). Two open anomalies on the CUDA harness — see the note below the table.

| Backend         | mean_latency (ms) | p95 (ms) | FPS  | RSS Δ (MB) | mAP50 (full) | mAP50-95 |
| --------------- | ----------------- | -------- | ---- | ---------- | ------------ | -------- |
| PyTorch (CUDA)  | 220.8             | 249.8    | 63.4 | 542.6      | 0.597        | 0.408    |
| ONNX FP32 CPU   | 2710.7            | 2926.1   | 5.16 | 218.7      | 0.597        | 0.408    |
| ONNX INT8 CPU   | 1765.7            | 2063.0   | 7.93 | 113.2      | 0.591        | 0.405    |
| ONNX FP32 CUDA  | 1084.1            | 1224.5   | 12.9 | 320.7      | 0.597        | 0.408    |
| ONNX INT8 CUDA  | 302.0             | 326.6    | 46.4 | 67.5       | 0.591        | 0.403    |

**Two diagnosed CUDA-harness findings** (ORT-gpu 1.26.0; all isolation experiments 2026-09-18):
1. **FP32-CUDA latency is 4.9× the PyTorch CUDA reference** (12.9 vs 63.4 FPS). Session-init logs show the Detect-head subgraph (`/model.22/` Gather/Concat/Unsqueeze/Mul chains — ops the CUDA EP *does* implement) is assigned to the CPU EP by ORT's "CPU execution path is deemed faster" cost heuristic, adding per-inference host↔device round-trips of the `(bs, 16, 8400)` head tensors. `session.disable_cpu_ep_fallback = 1` cannot be used as a remedy — session creation fails on this graph (CPU-assigned nodes remain). The FP32/INT8 CUDA latency rows are current-harness numbers, not the format's ceiling; the PyTorch CUDA reference is unaffected.
2. **INT8-on-CUDA box-coordinate divergence** — the root of the 98% detection-mode gate failure. Same input, same INT8 model, CPU vs CUDA session: cls channels agree (mean |Δ| ≈ 0, max 0.54) so the *detection sets* agree (counts match per image), but box coordinates diverge (max 175 px on the 640-letterbox scale) — greedy IoU-≥0.5 matching then collapses (21 matched pairs, mean IoU 0.088, recall gates fail). The INT8 artifact itself is healthy: through the CPU pipeline it shows the normal profile (31.9% gate failures, mean IoU 0.959, 185 pairs) and full-val mAP 0.403. Additionally, the INT8 session carries ~45 `MemcpyFromHost` nodes — one per conv bias `DequantizeLinear` (the CUDA EP has no kernel for 1-D/per-channel bias DQ in this build), the known QDQ-on-CUDA structural cost. Classification: an ORT 1.26.0-gpu QDQ/CUDA issue in this configuration; INT8 inference is validated on the CPU EP, and re-validating CUDA INT8 against a newer ORT-gpu is future work.

mAP columns are per-class means (`results/<backend>_perclass.csv`); ultralytics' instance-weighted "all" row for PT/FP32 is **0.601 / 0.411** — identical to the training-side record ([TRAINING.md §6](TRAINING.md#6-results)), and the FP32 export reproduces the PT per-class table exactly.

**PT ↔ ONNX FP32** (tensor mode, 91 images, `--atol 1e-4 --rtol 1e-3`): **PASS** — max_diff mean ≈ 0.002, cosine ≈ 1.000 (`results/consistency_report_tensor.json`).

**FP32 ↔ INT8** (detection mode, 91 images, CPU harness): **FAIL at the per-image gates** — 63/91 images pass (69%). Aggregate: mean IoU 0.959, mean score diff 0.032, mean recall 0.924, advisory cosine 0.9988. Failure breakdown (an image can trip several): recall < 0.97 on 18, score diff > 0.08 on 9, IoU < 0.92 on 8, class match on 3, one image lost its only detection — yet full-val mAP50-95 drops just 0.0026 (0.408 → 0.405). The gates are stricter than the task requires; see the measured note in §7.1. On the CUDA harness the same comparison is **anomalous** (98% fail, mean IoU 0.088) — diagnosed as the INT8-on-CUDA box-coordinate divergence (finding 2 above), not a model defect.

**PT ↔ ONNX FP32** (tensor mode) passes on both harnesses: CPU — max_diff mean ≈ 0.002, cosine ≈ 1.000 (`results/consistency_report_tensor.json`); CUDA — max_diff ≈ 0.002, cosine ≈ 1.000.

> The exact numbers depend on CUDA / cuDNN / ORT versions and dataset. The **budget** the pipeline enforces: INT8 should be 1.5–2× faster than FP32 on the same provider (**measured 1.54×** on the CPU EP; 3.59× on the CUDA harness, but the FP32-CUDA denominator is fallback-degraded — finding 1) with no more than a 0.01 mAP50-95 drop (**measured 0.0026 CPU / 0.0046 CUDA**); detection-mode consistency is judged by the §7 gates, which the CPU run fails on 31% of images while end-task accuracy holds — gate recalibration is future work (§7.1).

---

## 12. Lessons learned

* **Tail latency matters more than mean.** A pipeline that's 50ms mean but 200ms p99 breaks a real-time camera loop. Benchmark reports p99.
* **The quantizer's `EnableSubgraph=True` extra option matters.** Without it, `quantize_static` falls back to per-op quantization on certain layers, and the fallback costs measurable mAP.
* **Pre-warm the ORT session, not torch.** The first ORT forward is materially slower than steady state, and on CUDA that's where you discover CUDA EP isn't actually active.
* **A host copy of the input throttles the CUDA path.** Feeding `tensor.cpu().numpy()` into a CUDA session serializes GPU work behind a host round-trip; IO Binding + DLPack keeps the input tensor on the device (§4.3).
* **Keep the whole Detect head in FP32, not just Sigmoid/Softmax.** Op-type exclusion of `Sigmoid/Softmax` only skips the activation *nodes* — the upstream cls conv stays QDQ-wrapped and the post-Sigmoid class scores collapse to zero (0 detections). Excluding the whole `/model.22/` head by name-prefix keeps the cls-logit path in FP32 end to end and recovers the cls scores (max Δ 0.046 in post-sigmoid score space — §5.5); the head is <5% of FLOPs so the speed cost is negligible.
* **Atomic JSON is the difference between 30 s and 30 min of debugging.** A mid-run crash leaves the partial findings instead of a half-written (or empty) report.
* **Tests that depend on models test the model, not the code.** Pure-Python tests run in CI in 2 seconds; if they were coupled to `.onnx` they'd be skipped locally and lie about the build status.

---

## 13. Operating instructions

See the [README Quick Start](../README.md#quick-start) for the full command sequence
(export → inspect → consistency → quantize → infer → benchmark) and the CLI reference;
run `./clean.sh` to reset generated artifacts.

---

## 14. Glossary

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
