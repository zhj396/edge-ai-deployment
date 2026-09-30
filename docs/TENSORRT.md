# TensorRT Deployment (`src/tensorrt_engine.py` + `src/tensorrt_build.py`)

TensorRT 10.4.0 is the 6th inference backend (after PyTorch, ONNX FP32/INT8,
OpenVINO, ORT-C++), NVIDIA-GPU-only. It is an **optional** dep — the base pin
set does not include `tensorrt` / `cuda-python`. The engine mirrors
`OpenVINOEngine`'s API (`infer` / `infer_frames` / `_run_batch` with an
explicit `nc=`), registered in `src/__init__.py::_LAZY`, reusing the shared
`src/preprocess` + `src/postprocess` core like every other Python backend.

## Why TensorRT for Edge AI / interviews

TensorRT 10.4.0 pinned for the **Kaggle Tesla T4 (sm_75)** environment where
`trtexec` is unavailable — which *forces* the programmatic Python-API build
path. That constraint is the interview story: the build path demonstrates
`trt.Builder` + `OnnxParser` + `OptimizationProfile` + `BuilderFlag.FP16/INT8`
+ the **QDQ-vs-calibrator INT8 story** (explicit QDQ reusing ORT's
head-excluded `quantize_onnx_to_int8` as the robust default; the
`IInt8EntropyCalibrator2` + `OBEY_PRECISION_CONSTRAINTS` ablation that
**does not hold** the head FP32 on T4 → QDQ is the defensible fix), and the
inference path speaks the **TRT-10 tensor-name API** (`set_tensor_address` + `execute_async_v3`; the legacy
index-based `set_binding_address` / `enqueueV2` was removed) over the
`cuda.bindings.driver` "lean" bindings (CUDA 12.x, no PyCUDA). These are the
NVIDIA-deployment skills DE/Edge-AI interviews probe.

## Install

```bash
# TensorRT is GPU-only — install the GPU base set first, then the TRT overlay.
pip install -r requirements-kaggle.txt
pip install -r requirements-tensorrt.txt
```

`requirements-tensorrt.txt` pins `tensorrt==10.4.0` and `cuda-python>=12.3.0`,
matching the CUDA-12 runtime train of the Kaggle T4 environment behind
`requirements-kaggle.txt` (`onnxruntime-gpu==1.26.0` is the last CUDA-12
build; the base image ships a CUDA-12.x torch).

## Three build paths

| Path | Command | When |
|------|---------|------|
| **A — Python API** (primary) | `python main.py tensorrt build --model models/yolov8s.pt --precision fp16` | Full control over profile/workspace/calibrator; the interview story. Accepts a `.pt` (auto-exports a TRT-friendly ONNX) or a `.onnx` |
| **A' — trtexec** | `bash scripts/build_trt_engines.sh [fp16|int8]` | Local Linux; more deterministic layer fusion. Falls back to Path A on failure |
| **B — Ultralytics native** | `python main.py tensorrt export --model models/yolov8s.pt --precision fp16` | Kaggle/Colab; one step, no `trtexec`. INT8 needs `--data data/data.yaml` |

All three produce the same raw `[B, 4+nc, 8400]` head that `TensorRTEngine`
consumes. The default INT8 Path-A build reuses `src/quantize.py::
quantize_onnx_to_int8` (the **shared** `CalibrationSampler` — stratified +
phash + ResNet50 farthest-first — builds the image list; the **shared**
`preprocess_imgs` letterbox is the calibration preprocessing) to produce a
head-excluded QDQ ONNX that TRT consumes with the INT8 flag and **no
calibrator**. The `--calibrator` ablation instead feeds the sampler's image
list to TRT's `IInt8EntropyCalibrator2` directly (does not hold the head FP32
on T4 — see below). Either way the calibration input distribution matches the
deployed path exactly.

> **⚠️ TRT-friendly ONNX + the Turing dynamic-shape limit.** Path A must
> build from a TRT-friendly ONNX (`opset 13`, `simplify=False`), **not** the
> project's canonical `yolov8s_fp32.onnx` (`opset 17` + `onnxsim`), whose
> simplified dynamic `Shape`+`Slice` subgraph is one trigger. But the
> **real, opset-independent cause** on Turing (sm_75, e.g. T4) is the
> **dynamic batch dim itself**: the DFL reshape `[-1,4,16,8400]` over a
> symbolic batch forces TRT 10.4 to materialize a `Shape`+`Slice` subgraph it
> **cannot lower** on Turing —
> `nbDims > Dims::MAX_DIMS — Could not find any implementation for node
> ONNXTRT_ShapeSlice_*`. opset/simplify don't help (verified). The fix on
> Turing is **`--static`**: export a static-batch=1 ONNX (no symbolic dim →
> no Shape subgraph → no failure) and build a batch=1 engine (no optimization
> profile). On Ampere+ (sm_80+) the default dynamic build works. The canonical
> ONNX stays canonical for ORT/OpenVINO. Passing `--model X.pt` auto-exports the
> TRT-friendly ONNX to `models/yolov8s_trt.onnx` (add `--static` for Turing).

```bash
# FP16 (Path A) on a T4 / Turing — static batch=1
python main.py tensorrt build --model models/yolov8s.pt --static --precision fp16

# On Ampere+ (sm_80+) — dynamic 1..max_batch
python main.py tensorrt build --model models/yolov8s.pt --precision fp16

# INT8 (Path A) — default QDQ path: reuses ORT's quantize_onnx_to_int8
# (head excluded) on the TRT-friendly ONNX; no TRT calibrator
python main.py tensorrt build --static --precision int8 \
    --model models/yolov8s.pt --output models/yolov8s_int8.engine

# INT8 (Path B) — Ultralytics calibrates from data.yaml automatically
python main.py tensorrt export --precision int8 --data data/data.yaml
```

### INT8 Detect-head FP32 exclusion

The TRT INT8 build keeps the `/model.22/` Detect head in FP32 (the Path-A
mirror of ORT invariant #3). Whole-net INT8 collapses the head's unbounded
box-decode accumulation — T4 measurement:

| build | `max_diff` (box coords, 0–640) | `fail_rate` | `detection_fail_rate` |
|---|---|---|---|
| INT8 **whole-net** (`--calibrator --no-exclude-head`) | 242–517 | 100% | **50%** |
| INT8 **calibrator + OBEY** (legacy `--calibrator`) | 311–555 | 100% | **66.67%** |
| INT8 **QDQ** (default — head FP32 by QDQ omission) | single digits | — | **~18.75%** (symmetric-backbone divergence, see §cliff below) |
| FP32 TRT vs FP32 ONNX (plumbing baseline) | ~0.001–0.004 | 0% | **0%** |

FP16-whole-net survives (`max_diff~15`, FP16 has the exponent range INT8
lacks), so head FP32 protection is gated on `int8` — FP16/FP32 builds skip it.

**Mechanism — explicit QDQ (default), not calibrator+OBEY.** The default INT8
build feeds TRT an **explicit-quantization QDQ ONNX** produced by reusing
`src/quantize.py::quantize_onnx_to_int8` (the exact `/model.22/`
head-exclusion scope as ORT invariant #3) **on the TRT-friendly ONNX**
(opset 13, no-simplify, static). TRT 10 consumes Q/DQ graphs natively: set
the INT8 flag, **no calibrator**, and the head (which carries no
QuantizeLinear/DequantizeLinear nodes) stays FP32 **by construction** — no
`layer.precision`, no `OBEY_PRECISION_CONSTRAINTS`. This is the robust
mechanism; the head-exclusion scope is shared verbatim with the ORT path.

> **TRT-compatible QDQ settings (not ORT's defaults).** ORT's QDQ defaults
> (QUInt8 asymmetric activations + INT32-quantized bias) are NOT TRT-importable:
> TRT warns it doesn't support UINT8 Q/DQ zero-points, **requires fully
> symmetric quantization (every QuantizeLinear/DequantizeLinear zero-point must
> be all zeros)**, and its `IDequantizeLayer::setPrecision` rejects the INT32
> bias DequantizeLinear ("can only run in kINT8/kFP8/kINT4"). So the TRT-bound
> `quantize_onnx_to_int8` call uses **`activation_type=QInt8` (signed)** +
> **`extra_options={"ActivationSymmetric": True, "QuantizeBias": False}`**
> (symmetric activations → zero-point=0; bias left FP32, which TRT folds
> natively). The head-exclusion scope (`/model.22/`) is unchanged — only the
> activation dtype, symmetry, and bias handling differ from the ORT-CPU path.

**Why not calibrator+OBEY?** The first TRT INT8 attempt mirrored ORT's head
exclusion with `IInt8EntropyCalibrator2` + `layer.precision=float32` +
`set_output_type` + `OBEY_PRECISION_CONSTRAINTS` on the parsed
`INetworkDefinition` (70 `/model.22/` layers pinned). On T4 / TRT 10.4 this
**does not hold**: the head-"protected" engine collapses *identically* to the
whole-net engine (311–555 / 66.67% vs 242–517 / 50% — an ablation wash; the
precision constraints don't bind under calibrator-based implicit
quantization). `OBEY_PRECISION_CONSTRAINTS` is set but the builder does not
honor it for the calibrator path; the TRT 10.4 engine inspector returns only
layer *names* (`{"Layers":[...],"Bindings":[...]}`, no per-layer precision
field), so runtime precision can't be confirmed via the inspector — the
conf-cliff gate is the ground truth. Explicit QDQ sidesteps the whole class
of problem.

```bash
# Default INT8 = QDQ path (head FP32 by QDQ omission, no calibrator)
python main.py tensorrt build --static --precision int8 \
    --model models/yolov8s.pt --output models/yolov8s_int8.engine

# Ablation: legacy calibrator + OBEY (collapses on T4 — interview narrative only)
python main.py tensorrt build --static --precision int8 --calibrator \
    --model models/yolov8s.pt --output models/yolov8s_int8_calib.engine

# Ablation-within-ablation: whole-net INT8 (reproduces detection_fail_rate=50%)
python main.py tensorrt build --static --precision int8 --calibrator \
    --no-exclude-head --model models/yolov8s.pt \
    --output models/yolov8s_int8_wholenet.engine
```

Path B (`export`) has no `network` object exposed — head exclusion is Path A
only, and Path B INT8 has no QDQ step, so it quantizes whole-net (collapse).

### Entropy calibration: host-side OOM (source-traced)

The QDQ quantization step reuses `src.quantize.quantize_onnx_to_int8`, whose
`method` defaults to **`MinMax`**. The obvious lever to tighten the symmetric
activation scales (and so reduce the ~18–33% recall loss) is `Entropy` (KL)
calibration, exposed via the `--calib-method {MinMax,Entropy}` flag. On the T4,
`Entropy` at the default 300 samples **OOM-crashed the Kaggle notebook even on
GPU (`--device`)** — and this is *not* a GPU-memory problem; it is ORT's
calibrator memory model, traced in the source:

- `onnxruntime/quantization/calibrate.py` :: `HistogramCalibrater.collect_data`
  (which `EntropyCalibrater` inherits) does `outputs = self.infer_session.run(
  None, inputs)` then **`self.intermediate_outputs.append(outputs)`** — it
  **caches every batch's intermediate tensors in host memory** and only reduces
  to a histogram *after the full loop*. So host RAM scales **linearly** with
  `max_cal_samples × num_tensors × tensor_size`.
- `InferenceSession.run()` **always returns host numpy arrays** regardless of
  provider; `augment_graph` makes every quantization-candidate tensor a graph
  *output*, so each batch fetches *all* intermediate activations to host.
  Routing the forward to GPU (`--device`) moves only the *forward* — it does
  **not** reduce host RAM (it even adds a D2H copy per batch).
- `MinMaxCalibrater`, by contrast, has `max_intermediate_outputs` +
  `clear_collected_data()` → **incremental flush** (min/max per batch, raw data
  discarded) — that's why MinMax never OOM'd.

There is **no `extra_options` flag** to keep activations on-device or enable an
incremental histogram flush for Entropy; doing so means forking
`HistogramCalibrater.collect_data`. The real, proportional lever is
`--max-cal-samples` (the cache is linear in sample count):

```bash
# Entropy retry — cut the sample count (KL finds a threshold from ~50 samples)
python main.py tensorrt build --static --precision int8 \
    --model models/yolov8s.pt --output models/yolov8s_int8_ent.engine \
    --calib-method Entropy --max-cal-samples 64
# then consistency vs the opset-13 FP32 (apples-to-apples):
python main.py consistency --model1 tensorrt:models/yolov8s_int8_ent.engine \
    --model2 models/yolov8s_trt.onnx --max-images 12 --batch-sizes 1
```

**Measured (T4, 64 samples, GPU):** ran clean (the OOM was gone with the
smaller cal set), and Entropy *did* help modestly vs MinMax (300, same
apples-to-apples 12-image set): `detection_fail_rate` **33.33% → 25.00%**
(4/12 → 3/12 — one fewer flip, within Poisson noise on n=12), `max_diff`
260–494 → 236–410, `cos` 0.993–0.999 → 0.996–0.998. The KL fit tightens the
symmetric activation scales → less backbone divergence (direction confirmed by
the raw-tensor improvement, not just the noisy 1-image flip delta). But it
does **not** break the symmetric-INT8 floor — 25% still >> FP16-whole-net's
~0% (`max_diff~15`). So: **MinMax is the practical default; Entropy is a
modest improvement if shipping INT8 anyway; FP16 remains the consistent T4
choice.** Failure mode still demotion-heavy (recall loss).
The **conf-cliff gate** (Consistency section below) is the ground-truth check for
*detection consistency*, but **not** for *"did the head stay FP32"* — those were
conflated earlier. The authoritative head-FP32 check is the pure ONNX diagnostic
`scripts/diag_qdq_head.py models/yolov8s_trt_int8.onnx` (counts Q/DQ nodes under
`/model.22/`; pure `onnx`, no TRT/GPU): head Q/DQ=0 ⟹ the head stayed FP32 by
QDQ omission. On the **default QDQ path** that is always the case (verified,
0 head Q/DQ nodes) — yet `detection_fail_rate` can still be nonzero (~18.75% on
T4), because the TRT-bound QDQ forces **symmetric activation quantization**
(~2× coarser than ORT's asymmetric `QUInt8` on YOLOv8's ReLU backbone) and that
backbone error propagates into the FP32 head's box-decode, flipping the cliff on
~18% of images (cos stays 0.988–0.998 — *not* collapse, which tanks cosine). So:
`detection_fail_rate>0` on a QDQ-path run ⟹ genuine symmetric-backbone INT8
divergence, **not** "the head didn't stay FP32." The calibrator path collapses at
50–66% with much lower cosine; the magnitude + cosine distinguish the two.
`detection_fail_rate→0%` ⟹ head FP32 **and** backbone divergence under the cliff
(the FP16-whole-net result, `max_diff~15`). 18.75% is the honest cost of
TRT-compatible symmetric INT8 — the defensible result, not a bug.

## Run inference

```bash
python main.py tensorrt run --model models/yolov8s_fp16.engine --imgs-input data
# --max-batch must match the profile the engine was built with (default 8)
```

A serialized `.engine` carries no ultralytics metadata, so class names come
from `data.yaml` (`--data`, default `data/data.yaml`); absent → `class_N`.

## Consistency (`python main.py consistency`)

The `tensorrt:` prefix selects the TRT backend — `ModelWrapper` parses it and
deserializes the engine **lazily on the first forward** (so constructing the
wrapper without TRT installed / without the engine staged does not blow up).
TRT consumes the already-preprocessed tensor like the `onnx` path (unlike
`ort_cpp:` which re-preprocesses from files).

```bash
# FP16 — tensor mode (auto: loose tolerances — FP16 is reduced precision)
python main.py consistency \
    --model1 models/yolov8s_fp32.onnx \
    --model2 tensorrt:models/yolov8s_fp16.engine --mode tensor

# INT8 — detection mode (auto: loose tolerances, conf-cliff gate active)
python main.py consistency \
    --model2 tensorrt:models/yolov8s_int8.engine --mode detection
```

The auto-tolerance selector (`cli/consistency_tol.py`) is path-stem based, so
`tensorrt:models/yolov8s_int8.engine` and `tensorrt:models/yolov8s_fp16.engine`
→ loose pair automatically (any INT8 *or* FP16 side is reduced precision).
The loose pair only relaxes the advisory tensor-mode allclose — FP16-vs-FP32
box coords diverge to single digits on a 0–640 scale, so strict allclose
cannot pass and should not; the **conf-cliff gate** (`utils/comparison.compare_tensors`)
stays the authoritative catcher for spurious-detection explosions.

**Dual verdict (raw-tensor vs detection-level).** Because FP16 raw allclose
structurally cannot pass at any safe tolerance, `compare_tensors` reports the
two tensor-mode verdicts **side by side** rather than merging them: `allclose_passed`
(raw-tensor parity, the strict gate) and `detection_consistent` (the conf-cliff
gate). `src/consistency.py` surfaces both per image — `detection: CONSISTENT
(cliff clean)` on a benign FP16/INT8 drift, `detection: INCONSISTENT (cliff flip)
← real-regression signal` on the FP32-CUDA explosion class — and prints two
fail rates per config. A typical FP16-vs-FP32 run shows:

```
Inconsistent | max_diff=12.43 | cos=1.00 | cls_max_diff=3.82e-03 | above_cliff 10→10 (promoted=0, demoted=0) | detection: CONSISTENT (cliff clean)
fail_rate=100.00% | detection_fail_rate=0.00%
```

i.e. raw tensors diverge (FAIL, the truth) **and** detections are consistent
(PASS, the other truth) — neither masks the other. This is the
engineering-honest posture, not "tune it until FP16 passes".

## Benchmark (`python main.py benchmark`)

```bash
python main.py benchmark \
    --model tensorrt_fp16:models/yolov8s_fp16.engine \
    --model tensorrt_int8:models/yolov8s_int8.engine
```

The `tensorrt*` rows emit two extra metrics alongside the cross-backend
end-to-end number: **`kernel_latency_ms` / `kernel_fps`** — the GPU forward
pass (H2D + `execute_async_v3` + D2H) free of Python NMS / letterbox / disk
I/O, via `TensorRTEngine.kernel_timed_forward`. This preserves the TRT
project's standout metric (kernel-only ≈ 1.5–3× faster than end-to-end on a
T4) without breaking end-to-end parity across backends (the other rows just
omit the two columns). mAP validation runs via the **backend-agnostic native
evaluator** (`utils/map_eval.evaluate_map` driving the engine's own `infer()`)
for `tensorrt*` rows — a serialized `.engine` is not loadable by Ultralytics
`YOLO().val()`, but the native evaluator drives it directly, so the rows get
real mAP50 / mAP50-95 on the same metric as every other Python-driven backend
(the NNCF-INT8 vs QDQ-INT8 mAP delta is visible). `trt_cpp*` rows get the same
treatment (in-process pybind11 → native mAP), so the C++-vs-Python comparison
is apples-to-apples on every metric.

## Architecture notes

* **Optional-dep guard.** `try: import tensorrt as trt; from cuda.bindings
  import driver as cuda` at module top; `tensorrt_available()` predicate;
  constructing `TensorRTEngine` without the wheels raises `ImportError`
  pointing at `requirements-tensorrt.txt` (never `AttributeError`). The
  model-free test suite asserts this contract.
* **Lean bindings, one CUDA dialect.** `cuda.bindings.driver` (CUDA 12.x)
  returns `(CUresult, *values)` for every call, normalised through
  `_cuda_call` / `_cuda_check` (folded into `src/tensorrt_engine.py` and
  reused by `src/tensorrt_build.py`). No PyCUDA.
* **Context-managed.** `cuDevicePrimaryCtxRetain` + per-thread
  `cuCtxPushCurrent`/`cuCtxPopCurrent`; `release()` tears the primary
  context, the user-created stream, and the device buffers down in
  deterministic order.
* **Dynamic batch, no zero-pad.** `set_input_shape` per forward resolves the
  optimization profile (1..`max_batch`); any count in range is accepted
  directly (unlike a static-batch OpenVINO IR whose DFL reshape constant is
  baked to a fixed batch). `_effective_batch` caps + sub-loops; `max_batch`
  is a constructor param (a deserialized engine doesn't expose its profile
  bounds).
* **Shared core.** `TensorRTEngine` consumes `preprocess_imgs` /
  `preprocess_frames` host numpy and calls `post_process(nc=len(
  class_names))`. The standalone TRT project's own preprocess / postprocess /
  sampler / export / benchmark / consistency modules were **dropped** in
  favor of the shared core — this backend is additive, not a fork.

## trt_cpp (in-process C++ pybind11 backend)

`trt_cpp` is the 7th backend — the **same** serialized `.engine`, driven by an
in-process C++ pybind11 accelerator (`cpp/tensorrt/`) instead of cuda-python.
It is **not** a subprocess + `.npy` exchange (the `ort_cpp` template): it
mirrors the Python `tensorrt:` path exactly — it consumes the already-
preprocessed tensor `x`, returns the raw `[bs, 4+nc, 8400]` head, and gets
native mAP + `kernel_latency_ms`. That is the point: the `trt_cpp*` vs
`tensorrt*` rows isolate the **C++-binding vs cuda-python overhead** on an
identical metric surface, which a subprocess path (lacking mAP + kernel
timing) could not show.

### Build

The module is optional and **OFF by default** — a box without TensorRT + CUDA +
pybind11 must not fail the `ort_cpp` build. TRT (~1.5GB) and CUDA are **not**
vendored (unlike ORT's committed tarball); the build requires a TRT install whose
version **matches the `tensorrt` wheel that builds the `.engine`** (TRT engines
are version-locked across major versions — a 10.4-built engine won't deserialize
in 11.1). The project pins TRT **10.4.0 / CUDA 12.6** (the Kaggle T4 CUDA-12.x runtime).

**Decoupled setup (the documented, reproducible flow).** Python deps and the C++
SDK are installed by *separate* tools so they never overlap:

1. **Python deps via pip** (`tensorrt==10.4.0` + `cuda-python>=12.3.0` — needed to
   *build* the `.engine`):
   ```bash
   pip install -r requirements-kaggle.txt
   pip install -r requirements-tensorrt.txt
   ```
2. **C++ build deps via the setup script** — OpenCV C++ dev (apt; the pip
   `opencv-python` wheel doesn't ship `OpenCVConfig.cmake`/headers) + the TRT
   10.4.0.26 cuda-12.6 tarball (download + extract + export `TENSORRT_ROOT`/
   `LD_LIBRARY_PATH`). It does **not** touch pip — it only *verifies* the
   installed `tensorrt` wheel matches the tarball. **Source it, don't `bash` it**,
   so the exports persist into your shell:
   ```bash
   source scripts/trt_cpp_setup.sh    # NOT `bash ...` — exports must persist
   ```
   Portable: OpenCV auto-installs via apt on Debian/Ubuntu (skip with
   `SKIP_OPENCV_INSTALL=1`); the TRT tarball defaults to `/kaggle/working` on
   Kaggle (the documented T4 target), else `$HOME` — override with
   `TENSORRT_INSTALL_DIR=/opt source scripts/trt_cpp_setup.sh`. If the NVIDIA
   download needs auth on your network, the script prints manual download steps
   instead of failing cryptically.

Why decoupled: the tarball wheel and `requirements-tensorrt.txt`'s `tensorrt==10.4.0`
are the *same* underlying version (10.4.0.26), so installing both is benign but
redundant — letting pip own Python and the script own the C++ SDK removes the
overlap entirely. The script's verify step guarantees the `.engine` (built by the
pip wheel) cross-loads in the C++ module (linked against the tarball libs).

> **apt's default TensorRT is now 11.x (CUDA 13)** — do NOT use it. An 11.x
> install won't deserialize a 10.4-built `.engine`, and its CUDA-13 deps conflict
> with the Kaggle environment's CUDA-12.x runtime. The tarball gives the exact pinned
> 10.4.0.26 / CUDA-12.6 + the C++ headers/`libnvinfer.so` dev symlink the pip
> wheel omits.

Then build (from the venv so pybind11 finds the CLI's interpreter — avoids a
Python-ABI mismatch):

```bash
cmake -S cpp -B cpp/build -DBUILD_TRT_CPP=ON
cmake --build cpp/build --target _trt_cpp
# Emits cpp/build/tensorrt/_trt_cpp<SOABI>.so (Linux) / .pyd (Windows).
```

The finder (`cpp/tensorrt/cmake/FindTensorRT.cmake`) searches `$ENV{TENSORRT_DIR}`
/ `$ENV{TENSORRT_ROOT}` **first** (an explicit override wins over a wrong-version
system install), then `/usr/lib/x86_64-linux-gnu`, `/usr/local/tensorrt`,
`/opt/TensorRT-10.4.0.26`, `C:/TensorRT`. Configure prints
`TensorRT found: 10.4.0.26` — if it prints 11.x, your `TENSORRT_ROOT` isn't set in
the cmake shell (re-source the script, see above). The target carries `BUILD_RPATH`
pointing at the found TRT `lib/`, so `libnvinfer.so.10` resolves at runtime even
from a non-standard prefix (the setup script's `LD_LIBRARY_PATH` is
belt-and-suspenders). `CMAKE_CUDA_ARCHITECTURES` defaults to `75` (Tesla T4);
override for Ampere+ (`-DCMAKE_CUDA_ARCHITECTURES=80`). The default
(`-DBUILD_TRT_CPP=OFF`) leaves the `ort_cpp` build untouched — that is the CI gate.

### Use

```bash
# Consistency: trt_cpp vs the FP32 ONNX reference (raw-tensor allclose + conf-cliff gate)
python main.py consistency --model1 models/yolov8s_fp32.onnx \
    --model2 trt_cpp:models/yolov8s_fp16.engine --device 0

# Benchmark: trt_cpp vs Python tensorrt on the SAME .engine (apples-to-apples)
python main.py benchmark --model trt_cpp_fp16:models/yolov8s_fp16.engine \
    --model tensorrt_fp16:models/yolov8s_fp16.engine --validation

# Inference via the C++ backend
python main.py tensorrt run --model models/yolov8s_fp16.engine \
    --imgs-input data --backend cpp
```

`TRT_CPP_PATH` (absolute path to the `.so`/`.pyd`) overrides the build-output
glob — the escape hatch when multiple Python versions are built side-by-side.

### Architecture

* **Forward-only accelerator.** `cpp/tensorrt/` deserialises the `.engine`
  (`createInferRuntime` + `deserializeCudaEngine`) and runs
  `enqueueV3` (the C++ TRT-10 API; the Python binding names it
  `execute_async_v3`) over the CUDA **Runtime** API
  (`cudaMalloc`/`cudaMemcpyAsync`/`cudaStreamSynchronize`/`cudaFree`). No
  preprocess, no postprocess, no NMS, no CLI/main — those stay Python
  (`src/tensorrt_cpp_engine.py::TensorRTEngineCpp` reuses `preprocess_imgs` /
  `post_process` like every backend). It links `ort_core_common` to honor the
  documented contract in `cpp/common/CMakeLists.txt:5-7`, and `nvinfer` only
  (no `nvonnxparser` — deserialize-only; building stays in Python
  `build_tensorrt_engine`).
* **TRT-10 tensor-name API only.** `getNbIOTensors`/`getIOTensorName`/
  `getTensorIOMode`/`getTensorShape`/`setTensorAddress`/`setInputShape`/
  `enqueueV3`. The legacy `set_binding_address`/`enqueueV2` was removed
  upstream — no fallback (same as the Python path). Note the C++ method is
  `enqueueV3`, NOT the Python binding's `execute_async_v3` (the standalone
  reference `yolov8s_trt_cpp` used the Python name — a latent bug that does not
  compile against real TRT 10.4; corrected here).
* **CUDA Runtime API, not Driver API.** The Runtime API auto-manages the
  primary context, so there is no `cuCtxPushCurrent`/`cuCtxPopCurrent` (the
  Python engine's push/pop is a Driver-API artifact via cuda-python lean
  bindings). Single context (no per-thread pool — the Python caller is
  single-threaded). `release()` tears stream + device buffers down; the primary
  context is freed at process exit.
* **Lazy module load.** `src.benchmark.resolve_trt_cpp_module()` finds + loads
  the `.so`/`.pyd` via `importlib` (never a top-level import), mirroring
  `resolve_ort_cpp_exe`. `trt_cpp_available()` probes it; constructing
  `TensorRTEngineCpp` without the module raises `ImportError` (never
  `AttributeError`). A Python-ABI mismatch (built for Py3.11, run under Py3.12)
  surfaces as `RuntimeError` with a rebuild hint.
* **Do not construct both backends simultaneously.** The Python `tensorrt:`
  path retains a primary context via the Driver API; the C++ `trt_cpp:` path
  uses the Runtime API's auto-managed primary context. They share the
  underlying context (refcounted, sequential), and the benchmark always calls
  `release()` between backends, so only one is live at a time.

## ONNX validation — layered, per-backend

The canonical export path (`python main.py export`) validates every ONNX
three ways in `validate_onnx_model`:

1. `onnx.checker.check_model` — structural integrity (graph legality, op
   signatures, topology).
2. `onnx.shape_inference.infer_shapes` — shape consistency, **non-fatal**
   (caught, run only for its raising side-effect on broken graphs, **never
   saved back to disk** — the on-disk ONNX TRT reads is unchanged).
3. ORT runtime forward probe — `select_providers(device)`, the same provider
   selection as `infer`, on a random input.

The TRT path uses a **TRT-specific layering** instead of re-running the ORT
probe:

1. `onnx.checker.check_model` — structural gate inside
   `build_tensorrt_engine` (pure, no CUDA/ORT, clearer error than TRT's
   parser).
2. TRT `OnnxParser.parse()` — TRT's own strict graph validator (fails fast
   with per-error messages).
3. `build_serialized_network` — the authoritative acceptance test.

The ORT runtime probe is **deliberately skipped** on the TRT path:
**ORT-pass ≠ TRT-build** (ORT and TRT accept different graphs — the
`nbDims > Dims::MAX_DIMS` failure ORT runs fine; only the TRT build surfaces
it), so an ORT probe can't catch the TRT-specific failure class and would give
false confidence. The layered posture (check_model + TRT parser + build) is
the defensible answer to "how did you validate the ONNX before TRT?" — and the
*reason* for the layering (per-backend validation, not a one-size probe) is the
stronger interview story than blindly running ORT.

## Interview talking points

1. Why TRT 10.4 (no `trtexec`) is *more* interview-valuable, not less: it
   forces the programmatic build path over the CLI shortcut.
2. The TRT-10 tensor-name API and why the legacy `enqueueV2` was removed.
3. The lean-bindings `(CUresult, *values)` tuple idiom vs PyCUDA; the
   `cuCtxPushCurrent`/`PopCurrent` per-thread context discipline.
4. INT8 quantization — why the **explicit QDQ path** (reuse ORT's
   `quantize_onnx_to_int8` head-excluded QDQ on the TRT-friendly ONNX; INT8
   flag, no calibrator; head FP32 by QDQ omission) is the robust default, and
   why the `IInt8EntropyCalibrator2` + `layer.precision`/`OBEY` alternative
   does NOT hold the head FP32 under calibrator-based implicit quantization
   on T4/TRT 10.4 (ablation wash: head-excluded == whole-net). The calibrator
   path is the ablation that *motivates* the QDQ fix — the strongest interview
   story. Calibrator preprocessing reuses the shared `CalibrationSampler` for
   the image list, and `preprocess_single` (per-image letterbox from
   `src/preprocess.py`, then `/255.0` to mirror `_assemble_batch`) for the
   per-image path — distinct from the QDQ path's `preprocess_imgs` because the
   calibrator is fed one image at a time.
5. Kernel-only vs end-to-end latency: why Python NMS can dominate a small
   model and make FP16 *look* slower than PyTorch end-to-end.
6. Layered ONNX validation: the canonical path runs `onnx.checker` +
   shape-inference (non-fatal) + an ORT runtime probe; the TRT path layers
   `onnx.checker` + TRT's own `OnnxParser` + the build, and *skips* the ORT
   probe because ORT-pass ≠ TRT-build — per-backend validation, not a
   one-size probe.
