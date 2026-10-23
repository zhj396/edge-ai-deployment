# TensorRT Deployment (`src/tensorrt_engine.py` + `src/tensorrt_build.py`)

TensorRT 10.16.1 is the NVIDIA-GPU-only inference backend. It is an **optional**
dep — the base pin set does not include `tensorrt` / `cuda-python`. The engine
mirrors `OpenVINOEngine`'s API (`infer` / `infer_frames` / `_run_batch` with an
explicit `nc=`), registered in `src/__init__.py::_LAZY`, reusing the shared
`src/preprocess` + `src/postprocess` core like every other Python backend.

## Pinned environment and TRT-10 API

TensorRT 10.16.1 pinned for the **Kaggle Tesla T4 (sm_75)** environment where
`trtexec` is unavailable — so the build path is the programmatic Python API
(`trt.Builder` + `OnnxParser` + `OptimizationProfile` + `BuilderFlag.FP16/INT8`
+ the QDQ-vs-calibrator INT8 handling: explicit QDQ reusing ORT's head-excluded
`quantize_onnx_to_int8` as the default; the `IInt8EntropyCalibrator2` +
`OBEY_PRECISION_CONSTRAINTS` ablation, whose precision constraints may not
bind the head FP32 — see the INT8 section), and the inference path speaks the
**TRT-10 tensor-name API** (`set_tensor_address` + `execute_async_v3`) over the
`cuda.bindings.driver` "lean" bindings (CUDA 12.x).

## Install

```bash
# TensorRT is GPU-only — install the GPU base set first, then the TRT overlay.
# The NVIDIA extra index is required: the PyPI `tensorrt` metapackage is an
# sdist whose setup.py can't run in restricted sandboxes (Kaggle).
pip install -r requirements-kaggle.txt
pip install --extra-index-url https://pypi.nvidia.com -r requirements-tensorrt.txt
```

`requirements-tensorrt.txt` pins `tensorrt==10.16.1.11` and `cuda-python>=12.3.0`,
matching the CUDA-12 runtime train of the Kaggle T4 environment behind
`requirements-kaggle.txt` (`onnxruntime-gpu==1.26.0` is the CUDA-12 build
pinned there; the base image ships a CUDA-12.x torch). The file header
documents how to re-pin if the pinned version stops resolving on NVIDIA's
index (query the index, pin another 10.x — the code targets the TRT-10
tensor-name API, stable across 10.x).

## Three build paths

| Path | Command | When |
|------|---------|------|
| **A — Python API** (primary) | `python main.py tensorrt build --model models/yolov8s.pt --precision fp16` | Full control over profile/workspace/calibrator. Accepts a `.pt` (auto-exports a TRT-friendly ONNX) or a `.onnx` |
| **A' — trtexec** | `bash scripts/build_trt_engines.sh [fp16\|int8]` (`STATIC=1` for the Turing static route) | Local Linux where `trtexec` is available. fp16 builds via `trtexec` with a Path-A fallback on failure; int8 routes directly to Path A (the QDQ head-exclusion pipeline is Python-API only) |
| **B — Ultralytics native** | `python main.py tensorrt export --model models/yolov8s.pt --precision fp16` | Kaggle/Colab; one step, no `trtexec`. INT8 needs `--data data/data.yaml` |

All three produce the same raw `[B, 4+nc, 8400]` head that `TensorRTEngine`
consumes. The default INT8 Path-A build reuses `src/quantize.py::
quantize_onnx_to_int8` (the **shared** `CalibrationSampler` — stratified +
phash + ResNet50 farthest-first — builds the image list; the **shared**
`preprocess_imgs` letterbox is the calibration preprocessing) to produce a
head-excluded QDQ ONNX that TRT consumes with the INT8 flag and **no
calibrator**. The `--calibrator` ablation feeds the sampler's image
list to TRT's `IInt8EntropyCalibrator2` directly (the precision constraints
may not bind the head FP32 — see below). Either way the calibration input
distribution matches the deployed path.

> **⚠️ TRT-friendly ONNX + the Turing dynamic-shape limit.** Path A builds
> from a TRT-friendly ONNX (`opset 13`, `simplify=False`). The project's
> canonical `yolov8s_fp32.onnx` (`opset 17` + `onnxsim`) carries a simplified
> dynamic `Shape`+`Slice` subgraph — one trigger of the limit. On the
> validated Turing T4 (sm_75, TRT 10.x) the dynamic-batch build fails at
> tactic selection; the paraphrased failure-log signature is
> `nbDims > Dims::MAX_DIMS — Could not find any implementation for node
> ONNXTRT_ShapeSlice_*` (node numbers elided). The cause is the dynamic
> batch dim itself: the DFL reshape
> `[-1,4,16,8400]` over a symbolic batch forces TRT to materialize a
> `Shape`+`Slice` subgraph it cannot lower on Turing — inferred from the
> failure log, not documented by NVIDIA. The Turing route is
> **`--static`**: it exports a static-batch=1
> ONNX (a concrete batch dim keeps the subgraph out of the graph) and builds
> a batch=1 engine (no optimization profile). On Ampere+ (sm_80+) the dynamic
> route is the expected default; its build verdict has no recorded run on
> the pinned environment (see [Provenance](#provenance)). The canonical
> ONNX stays canonical for ORT/OpenVINO. Passing `--model X.pt` auto-exports the
> TRT-friendly ONNX to `models/yolov8s_trt.onnx`; a `--static` build exports to
> `models/yolov8s_trt_static.onnx`, so the two cached exports never collide
> (the same split `scripts/build_trt_engines.sh` uses).

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
python main.py tensorrt export --model models/yolov8s.pt --precision int8 \
    --data data/data.yaml
```

### INT8 Detect-head FP32 exclusion

The TRT INT8 build keeps the `/model.22/` Detect head in FP32 (the Path-A mirror of
the ORT INT8 path's head exclusion). Quantizing the head to INT8 breaks the head's
unbounded box-decode accumulation.

> **Provenance.** Measured figures below come from a full recorded run on the pinned
> environment: Kaggle Tesla T4 (sm_75), `tensorrt==10.16.1.11` / cuda-python
> (CUDA-12.x), Python 3.13, imgsz 640, batch 1, the `--static` Turing route.
> Consistency runs: the default `--max-images 20` budget → 16 seed-42-sampler images
> (12-class stratified quotas) — the same set the recorded OpenVINO
> (`docs/OPENVINO.md`) and ORT-C++ (`docs/ORT_CPP.md`) runs used; the FP32-TRT
> baseline pins strict (1e-4/1e-3) explicitly because the `.engine` suffix routes the
> auto-selector loose. **Run artifacts, archived locally outside the repo**: per-step
> logs (`01`–`06_*.log`), the 10 consistency JSON reports (05a/05b ran tensor mode
> only), the benchmark summary + per-class CSVs (`benchmark_summary_<TS>.csv` +
> `*_perclass_*.csv`), the engine-inspector dump, `env.txt` (versions + `nvidia-smi`),
> and `app.log` with the full per-image lines. The mechanism structure — why each lever
> exists and which builds fail/survive — matches the figures.

The consistency-run structure (the detection fail share is the share of images whose
detection-level verdict flipped — recorded per image by the consistency harness; all
six builds ran the same 16-image set):

| build | `max_diff` (box coords, 0–640) | tensor `fail_rate` | detection fail share |
|---|---|---|---|
| INT8 **whole-net** (`--calibrator --no-exclude-head`) | 240.3–547.8 | 100% | **37.50%** (6/16; three catastrophic — `8→0`, `12→0`, `39→0`) |
| INT8 **calibrator + OBEY** (`--calibrator`) | 250.8–568.0 | 100% | **43.75%** (7/16 — six of the seven are whole-net's flip set: OBEY is a no-op under calibrator-based implicit quantization, up to one borderline image) |
| INT8 **QDQ** (default — head FP32 by QDQ omission) | 261.3–510.5 | 100% | **0%** on the conf-cliff gate (16/16 `CONSISTENT (cliff clean)`; per-image IoU/score gates in detection mode: 25.00% — the symmetric-backbone cost, see the conf-cliff discussion under Consistency) |
| FP32 TRT vs FP32 ONNX (plumbing baseline) | strict allclose, 0 failures | 0% | **0%** |

FP16-whole-net keeps detections consistent (`max_diff` 3.7–12.9 on the same
run, cosine ≥ 0.999999, `detection_fail_rate=0%` — FP16 has the exponent
range INT8 lacks), so head FP32 protection is gated on `int8` — FP16/FP32
builds skip it.

**Mechanism — explicit QDQ (default); calibrator+OBEY is the ablation.** The
default INT8 build feeds TRT an **explicit-quantization QDQ ONNX** produced
by reusing `src/quantize.py::quantize_onnx_to_int8` (the exact `/model.22/`
head-exclusion scope as the ORT INT8 path) **on the TRT-friendly ONNX**
(opset 13, no-simplify, static). TRT 10 consumes Q/DQ graphs natively: set
the INT8 flag, **no calibrator**, and the head (which carries no
QuantizeLinear/DequantizeLinear nodes) stays FP32 **by construction** — no
`layer.precision`, no `OBEY_PRECISION_CONSTRAINTS`. The head-exclusion scope
is shared verbatim with the ORT path.

> **TRT-compatible QDQ settings.** TRT warns it doesn't support UINT8 Q/DQ
> zero-points, **requires fully symmetric quantization (every
> QuantizeLinear/DequantizeLinear zero-point must be all zeros)**, and its
> `IDequantizeLayer::setPrecision` rejects the INT32 bias DequantizeLinear
> ("can only run in kINT8/kFP8/kINT4"). The TRT-bound
> `quantize_onnx_to_int8` call therefore uses **`activation_type=QInt8`
> (signed)** + **`extra_options={"ActivationSymmetric": True,
> "QuantizeBias": False}`** (symmetric activations → zero-point=0; bias left
> FP32, which TRT folds natively). The head-exclusion scope (`/model.22/`)
> is the same one the ORT INT8 path uses; the activation dtype, symmetry,
> and bias handling are the TRT-compatible settings.

**Why calibrator+OBEY is the ablation.** `IInt8EntropyCalibrator2` +
`layer.precision=float32` + `set_output_type` + `OBEY_PRECISION_CONSTRAINTS`
on the parsed `INetworkDefinition` (the `/model.22/` layer set pinned) pins
the head FP32 only where the constraints bind at tactic selection — under
calibrator-based implicit quantization they may not bind, and the head then
runs INT8 with the whole-net detection-consistency cost the calibration run
measures (pinned-environment result: `detection_fail_rate` **43.75%** vs
whole-net's **37.50%** — six of the seven cliff-flip images are shared with
the whole-net build, at the same max_diff band; per-image IoU/score gates
68.75% both. Pinning 70 head layers and excluding nothing yield (up to one
borderline image) the same failure mode on the same images — per-layer
precision is a no-op under calibrator-based implicit quantization).
`OBEY_PRECISION_CONSTRAINTS` is set; whether the builder honors it for the
calibrator path is build-dependent. On the TRT builds whose engine inspector
dump carries only layer *names* (`{"Layers":[...],"Bindings":[...]}`, no
per-layer precision field — verified on the pinned 10.16.1 build: 145
parsed layers / 61 head layers, zero precision tokens —
`scripts/diag_trt_head.py` tallies per-layer precision when a build's dump
exposes it), runtime precision can't be confirmed via the inspector — the
conf-cliff gate is the ground truth. The default QDQ path is explicit
quantization: TRT reads the scales from the graph, with no calibrator in
the loop.

```bash
# Default INT8 = QDQ path (head FP32 by QDQ omission, no calibrator)
python main.py tensorrt build --static --precision int8 \
    --model models/yolov8s.pt --output models/yolov8s_int8.engine

# Ablation: calibrator + OBEY (the precision constraints may not bind
# the head FP32 — see the INT8 section)
python main.py tensorrt build --static --precision int8 --calibrator \
    --model models/yolov8s.pt --output models/yolov8s_int8_calib.engine

# Whole-net INT8 (no head exclusion; the ablation row — expected to degrade
# detections, see the comparison table)
python main.py tensorrt build --static --precision int8 --calibrator \
    --no-exclude-head --model models/yolov8s.pt \
    --output models/yolov8s_int8_wholenet.engine
```

Path B (`export`) has no `network` object exposed — head exclusion is Path A
only, and Path B INT8 has no QDQ step, so it quantizes whole-net.

### Entropy calibration: host-side memory model (ORT calibrator)

The QDQ quantization step reuses `src.quantize.quantize_onnx_to_int8`, whose
`method` defaults to **`MinMax`**. The candidate lever to tighten the symmetric
activation scales is `Entropy` (KL) calibration, exposed via the
`--calib-method {MinMax,Entropy}` flag. Entropy's host-memory footprint is set
by ORT's calibrator design, not by the compute device — the calibration loop
holds every batch's intermediate tensors in host memory until the full loop
completes:

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
- `MinMaxCalibrater` has `max_intermediate_outputs` + `clear_collected_data()`
  → **incremental flush** (min/max per batch, raw data discarded), which keeps
  its host footprint flat in the sample count.

There is **no `extra_options` flag** to keep activations on-device or enable an
incremental histogram flush for Entropy; doing so means forking
`HistogramCalibrater.collect_data`. The proportional lever is
`--max-cal-samples` (the cache is linear in sample count):

```bash
# Entropy at a reduced sample count (the KL threshold search needs a few dozen)
python main.py tensorrt build --static --precision int8 \
    --model models/yolov8s.pt --output models/yolov8s_int8_ent.engine \
    --calib-method Entropy --max-cal-samples 64
# then consistency vs the opset-13 FP32:
python main.py consistency --model1 models/yolov8s_fp32.onnx \
    --model2 tensorrt:models/yolov8s_int8_ent.engine --mode tensor --device 0
```

**Measured (T4, Entropy-64 vs MinMax-300, same 16-image set):**
Entropy-64 — `detection_fail_rate` 25.00% (4/16 cliff flips, none
catastrophic: the worst is `12→3` and `39→31`, versus the calibrator builds'
`8→0`/`39→0`), per-image IoU/score gates 25.00% (4/16, all count/score
edge failures — one at IoU 1.0000 with count+2; class match 1.000 everywhere,
no full-mismatch images), cosine 0.993–0.999.
MinMax-300 (the 5c QDQ row) — cliff gate 0%, per-image IoU/score gates
25.00% (4/16, including an IoU-0.81 box-shift failure). Reading:
Entropy's cliff rate is nominally worse on n=16 but its flips are an order
of magnitude lighter (threshold-edge partial loss vs whole-detection
collapse), and the release-relevant per-image gate lands at the same rate
with lighter failure forms (count/score edges vs box shifts).
Configuration guidance: **MinMax is the default; Entropy is the lever if
shipping INT8 anyway; FP16 is the default T4 build precision (its
detection-consistency figure: `detection_fail_rate=0%` on the recorded
run).** Failure mode: demotion-heavy (recall loss), bounded by the
Entropy fit — flips stay at the threshold edge; the image's above-cliff set
does not collapse.
The **conf-cliff gate** (Consistency section below) checks
*detection consistency*; the head-FP32 question is a separate check — the two
questions are distinct. The head-FP32 check is the pure ONNX diagnostic
`scripts/diag_qdq_head.py models/yolov8s_trt_static_int8.onnx
models/yolov8s_trt_static.onnx` (counts Q/DQ nodes under
`/model.22/`; pure `onnx`, no TRT/GPU): head Q/DQ=0 ⟹ the head stayed FP32 by
QDQ omission. On the **default QDQ path** that holds by construction and on
the pinned build (654 nodes, 393 Q/DQ — all in the backbone — 90 head nodes,
**zero head Q/DQ**; the FP32 reference export: 261 nodes, 0 Q/DQ, the same
90 head nodes) — yet the per-image IoU/score gates can still fail
(25.00% on the recorded run), because the TRT-bound QDQ forces
**symmetric activation quantization**, whose coarser activation scales let
backbone error propagate into the FP32 head's box-decode and shift boxes /
drift scores past the strictest gates (cosine stays 0.994–0.998; a collapse
is the signature that tanks cosine). So: a nonzero per-image gate failure on
a QDQ-path run points to symmetric-backbone INT8 divergence; the head
staying FP32 is verified by the diagnostic above. The calibrator path fails
with a much larger `max_diff` alongside cliff flips and lower per-image
detection quality; magnitude + the gate verdicts distinguish the
classes. A zero detection fail share ⟹ head FP32 **and** backbone divergence
under the cliff (the FP16-whole-net case).

## Run inference

```bash
python main.py tensorrt run --model models/yolov8s_fp16.engine --imgs-input data
# --max-batch caps forwards (default 8); the engine also reads the profile
# max from the serialized engine, and a static engine steps at 1.
```

A serialized `.engine` carries no ultralytics metadata, so class names come
from `data.yaml` (`--data`, default `data/data.yaml`); absent → `class_N`.

## Consistency (`python main.py consistency`)

The `tensorrt:` prefix selects the TRT backend — `ModelWrapper` parses it and
deserializes the engine **lazily on the first forward** (so constructing the
wrapper without TRT installed / without the engine staged does not fail).
TRT consumes the already-preprocessed tensor like the `onnx` path (`ort_cpp:`
is the backend that re-preprocesses from files).

```bash
# FP16 — tensor mode (loose tolerances: FP16 is reduced precision)
python main.py consistency \
    --model1 models/yolov8s_fp32.onnx \
    --model2 tensorrt:models/yolov8s_fp16.engine --mode tensor \
    --atol 1e-3 --rtol 1e-2

# INT8 — detection mode (per-image IoU/class/score gates; the tensor-mode
# conf-cliff gate does not run in detection mode)
python main.py consistency \
    --model2 tensorrt:models/yolov8s_int8.engine --mode detection
```

For reduced-precision pairs (FP16/INT8) the consistency CLI auto-selects
the loose pair (`atol 1e-3`, `rtol 1e-2`) from the model-pair stems, sized
for reduced-precision numerical noise — the explicit flags in the command
above pin the same pair. At FP16 precision raw box coords are not expected
to pass strict allclose at any safe tolerance (recorded on the pinned
environment: `max_diff` 3.7–12.9 on a 0–640 scale across the 16-image
set — FP16's ~3-digit mantissa on the head's unbounded box-decode
accumulation);
the **conf-cliff gate**
(`utils/comparison.compare_tensors`, wired to the deployed conf threshold
in `src/consistency.py`) stays the authoritative gate for spurious-detection
count increases. Note the device flag's semantics on a `tensorrt:` pair: a
numeric id (`--device 0`) selects the TRT engine's GPU while the reference
model, sampler, and preprocessing run on CPU — `cuda` is the setting that
puts the reference model on GPU.

**Dual verdict (raw-tensor vs detection-level).** Because FP16 raw allclose
is not expected to pass at any safe tolerance (recorded divergence: the
`max_diff` 3.7–12.9 band above), `compare_tensors` reports the two
tensor-mode verdicts as separate fields: `allclose_passed` (raw-tensor
parity, the strict gate) and `detection_consistent` (the conf-cliff gate).
`src/consistency.py` surfaces both per failing batch (`--batch-sizes 1`
degenerates to per image) — `detection: CONSISTENT (cliff clean)` on a
benign FP16/INT8 drift, `detection: INCONSISTENT (cliff flip)` on the
FP32-CUDA cliff-flip class. The per-batch entries in `detailed_results`
carry the two verdict fields (`allclose_passed` / `detection_consistent`);
the per-config summary aggregates them into two fail rates (`fail_rate` /
`detection_fail_rate`, plus `detection_failures`) and the `*_mean`
statistic fields. The per-batch log line format (the `fail_rate` line is
the per-config summary, logged once after the config's batches complete):

```
Inconsistent | max_diff=<max_diff> | cos=<cos> | cls_max_diff=<cls_max_diff> | above_cliff <n>→<n> (promoted=<p>, demoted=<d>) | detection: CONSISTENT (cliff clean)
```

The per-config summary line, after the config's batches:

```
fail_rate=<f>% | detection_fail_rate=<d>%
```

i.e. raw tensors diverge (allclose-FAIL) **and** detections are consistent
(detection-PASS) — neither masks the other, so the raw divergence does not
have to be tuned away to read the detection verdict. On the recorded run
this printed `fail_rate=100.00% | detection_fail_rate=0.00%` on all 16
images for the FP16 engine (`above_cliff X→X, promoted=0, demoted=0` per
image).

**Reading the two `detection_fail_rate` fields.** The field's semantics are
mode-dependent (the conf-cliff gate is tensor-mode only —
`src/consistency.py` notes this): in tensor mode it is the share of images
that flipped the conf-cliff gate; in detection mode the log line's
`detection_fail_rate` is always 0 and the per-image IoU/class/score gate
results live in `fail_rate` plus the report's `per_image_detection` /
`det_*` fields. On the recorded run the INT8-QDQ pair read: tensor mode
`fail_rate=100% | detection_fail_rate=0%`, detection mode
`fail_rate=25.00%` (4/16 IoU/score-gate failures, class match 1.000
everywhere). Three numbers — raw allclose, conf-cliff, per-image gates —
answer three different questions; read them pairwise, never collapsed.

## Benchmark (`python main.py benchmark`)

```bash
python main.py benchmark \
    --model tensorrt_fp16:models/yolov8s_fp16.engine \
    --model tensorrt_int8:models/yolov8s_int8.engine
```

The `tensorrt*` rows fill two extra summary-CSV columns alongside the
cross-backend end-to-end number: **`Kernel_latency_ms` / `Kernel_fps`** — the
GPU-side forward window timed at stream sync (GPU work already retired before
the timer starts does not re-appear in it), free of Python NMS / letterbox /
disk I/O, via `TensorRTEngine.kernel_timed_forward` (other rows leave the two
columns empty; the end-to-end latency/FPS columns stay comparable across all
rows). `Peak_Memory_MB` on `tensorrt*` rows covers torch-visible allocations
only — the engine's cuda-python allocations are outside torch's stats — so
`RSS_Increase_MB` is the host-side memory figure to read on those rows.
The TRT GPU id comes from the `TENSORRT_DEVICE` env var (default `0`),
mirroring `OPENVINO_DEVICE` for the OpenVINO backends — `--device` still
controls only where the preprocessing runs. The pin works by setting
`CUDA_VISIBLE_DEVICES`, which CUDA reads once at process init: it is
effective only when the TRT backend is the first CUDA initialization in
the process (a single-TRT-backend run, or a TRT-only `consistency` run).
The engine detects an inert pin after `cuInit` (more than one device
visible under the pin means an earlier backend initialized CUDA first):
it warns, runs on physical GPU 0, and reports the id it actually executes
on through `engine.device_id` — `benchmark` records that id in the row's
`Device` column. mAP validation runs via the **backend-agnostic native
evaluator** (`utils/map_eval.evaluate_map` driving the engine's own `infer()`)
— the same evaluator every Python-driven backend uses; a Path-A `.engine`
carries no Ultralytics metadata for `YOLO().val()` to key on, so the rows
carry mAP50 / mAP50-95 on the same metric as every other Python-driven
backend (the NNCF-INT8 vs QDQ-INT8 mAP delta is visible).

**Recorded run (batch 1; latency columns on the 16-image speed set at the canonical
protocol — `--warmup 10 --runs 25` (the CLI defaults); mAP columns on the full
397-image val split via `--validation`):**

| backend | FPS | p50 (ms) | Kernel_latency_ms | Kernel_fps | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| `onnx_fp32` (CPU EP) | 5.58 | 2498 | — | — | 0.590 | 0.402 |
| `tensorrt` (FP32 engine) | 68.36 | 202 | 0.0077 | 130k | 0.590 | 0.402 |
| `tensorrt_fp16` | 79.71 | 174 | 0.0078 | 128k | 0.591 | 0.403 |
| `tensorrt_int8` | 80.59 | 172 | 0.0082 | 122k | 0.593 | 0.395 |

Reading the table: the FP32 TRT row's mAP matches the ONNX FP32 row
**per class** — 11 of 12 classes identical on both metrics, chair
within 2e-7 — a second, metric-surface-independent confirmation of the
FP32 engine's numerical fidelity (the first is the strict consistency
baseline). FP16's mAP delta (+0.001 / +0.001) is the mAP-surface payoff
of its benign dual verdict; INT8-QDQ's (+0.003 / −0.007 on mAP50-95) sits
at the level the per-image gate cost the consistency run measured. The
end-to-end columns sit almost entirely on the host: the GPU forward is
~0.008 ms against a 172–202 ms sweep, so e2e FPS reflects NMS / letterbox
/ I/O — FP16 gains ~17% e2e over the FP32 engine, while FP16→INT8 is flat
e2e. The kernel columns' run-to-run spread on this card is comparable to
the INT8-vs-FP16 gap, so the durable T4 statements are: INT8 wins model
size and compute headroom, the host pipeline dominates e2e at this scale,
and only the kernel columns can see GPU-side differences at all. (The
`onnx_fp32` row runs on the CPU execution provider — the default
`--device` — so its timing is indicative against the GPU rows; its
per-image detection counts agree with the FP32-TRT row on 396 of 397
images, the remaining image by a single detection.)

## Architecture notes

* **Optional-dep guard.** `try: import tensorrt as trt; from cuda.bindings
  import driver as cuda` at module top; `tensorrt_available()` predicate;
  constructing `TensorRTEngine` without the wheels raises `ImportError`
  pointing at `requirements-tensorrt.txt` (never `AttributeError`). The
  model-free test suite asserts this contract.
* **Lean bindings, one CUDA dialect.** `cuda.bindings.driver` (CUDA 12.x)
  returns `(CUresult, *values)` for every call, normalised through
  `_cuda_call` / `_cuda_check` (folded into `src/tensorrt_engine.py` and
  reused by `src/tensorrt_build.py`).
* **Context-managed.** The primary context is adopted via
  `cuDevicePrimaryCtxRetain` and pushed/popped per thread around init,
  forward, and release; `release()` tears down the user-created stream, the
  device buffers, and finally the primary context, popping only a context it
  pushed.
* **Dynamic batch.** `set_input_shape` per forward sets the actual batch
  count; every count in the profile range (1..`max_batch`) runs directly.
  `_effective_batch` caps the requested count (the `infer` /
  `infer_frames` loops step over forwards at it); the effective `max_batch`
  is the constructor value capped by the serialized engine's
  optimization-profile max, and a static engine (no profile) steps at 1.
* **Shared core.** `TensorRTEngine` consumes `preprocess_imgs` /
  `preprocess_frames` host numpy and calls `post_process(nc=len(
  class_names))`. Preprocess / postprocess / sampler / export / benchmark /
  consistency all run through the repo's shared modules — this backend is
  additive on the shared core.

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

The TRT path uses a **TRT-specific layering**:

1. `onnx.checker.check_model` — structural gate in `_build_engine`
   (invoked by `build_tensorrt_engine` after the CUDA context is pushed;
   the check itself is pure `onnx`).
2. TRT `OnnxParser.parse()` — TRT's own strict graph validator (fails fast
   with per-error messages).
3. `build_serialized_network` — the authoritative acceptance test.

The ORT runtime probe is **skipped** on the TRT path: ORT and TRT accept
different graphs (the `nbDims > Dims::MAX_DIMS` failure builds fine under
ORT), and the TRT-specific failure class surfaces only at
`build_serialized_network`. The layered posture (check_model + TRT parser +
build) validates each backend through its own acceptance path.

## Design notes

1. The missing `trtexec` on the pinned T4 environment is why the build path is
   the programmatic Python API (`trt.Builder` + `OnnxParser` +
   `OptimizationProfile` + `BuilderFlag.FP16/INT8`).
2. The TRT-10 tensor-name API (`set_tensor_address` + `execute_async_v3`) as
   the engine's execution contract.
3. The lean-bindings `(CUresult, *values)` tuple idiom; the
   `cuCtxPushCurrent`/`PopCurrent` per-thread context discipline.
4. INT8 quantization — the **explicit QDQ path** (reuse ORT's
   `quantize_onnx_to_int8` head-excluded QDQ on the TRT-friendly ONNX; INT8
   flag, no calibrator; head FP32 by QDQ omission) is the default, and the
   `IInt8EntropyCalibrator2` + `layer.precision`/`OBEY` alternative may not
   bind the head FP32 at tactic selection under calibrator-based implicit
   quantization (head-excluded == whole-net in the ablation). Calibrator
   preprocessing reuses the shared `CalibrationSampler` for the image list,
   and `preprocess_single` (per-image letterbox from `src/preprocess.py`,
   then `/255.0` to mirror `_assemble_batch`) for the per-image path —
   preprocessing runs per image, and `get_batch` then hands TRT a
   `calib_bs`-sized batch (capped by `--max-batch`), so the per-image
   entry point is the matching one for building each batch.
5. Kernel-only vs end-to-end latency: the kernel-only window excludes Python
   NMS, which can dominate a small model's end-to-end time.
6. Layered ONNX validation: the canonical path runs `onnx.checker` +
   shape-inference (non-fatal) + an ORT runtime probe; the TRT path layers
   `onnx.checker` + TRT's own `OnnxParser` + the build, and *skips* the ORT
   probe because ORT and TRT accept different graphs — each backend is
   validated through its own acceptance path.
