# OpenVINO Deployment

Intel's OpenVINO toolkit is the runtime for Intel CPUs, Intel integrated GPUs, Intel Arc GPUs, and
(on older OpenVINO releases) Intel Movidius VPUs. This toolchain supports it as an additional
inference backend alongside PyTorch, ONNX FP32/INT8, and TensorRT.

## Why OpenVINO for Edge AI

| Hardware | Best backend | Why |
|---|---|---|
| Intel Xeon / Core CPU | **OpenVINO** | oneDNN + VNNI tuning (the validated i5-13420H exposes AVX-VNNI and no AVX-512 — `/proc/cpuinfo` flags); the INT8-vs-FP16 gain on CPU is measured in the table under Measured Performance |
| Intel iGPU / Arc dGPU | **OpenVINO** | OpenVINO GPU plugin; requires an Intel GPU driver (see the device-preflight note below) |
| Movidius VPU (USB stick / RPi AI Kit) | — | The MYRIAD plugin was removed from recent OpenVINO releases and is absent from this repo's pinned `openvino==2026.3.0` |
| NVIDIA Jetson / dGPU | ONNX Runtime (CUDA) | OpenVINO targets Intel hardware; NVIDIA is served by ONNX Runtime's CUDA execution provider |
| ARM Cortex-A | ONNX Runtime | OpenVINO ARM support exists but limited |

OpenVINO targets Intel CPU / iGPU / Arc hardware; NVIDIA hardware is served by
ONNX Runtime's CUDA execution provider in this toolchain.

## Install

```bash
# Pinned, reproducible set (openvino==2026.3.0 + nncf==3.3.0) — the exact
# versions the converter / engine / NNCF-INT8 path are validated against.
# Install ON TOP OF requirements-cpu.txt (torch/ort live there).
pip install -r requirements-cpu.txt
pip install -r requirements-openvino.txt

# Minimal single-package installs (NOT pinned — may drift from validated set):
#   pip install openvino        # runtime only (smaller, no convert CLI)
#   pip install nncf            # adds NNCF INT8 quantization
```

These are **optional** dependencies — the rest of the toolchain works without
them: the modules import cleanly, and constructing `OpenVINOEngine` or calling
the convert/quantize entry points raises `ImportError` with an install hint
when the packages are absent.

## Convert ONNX → OpenVINO IR

```bash
python main.py openvino convert \
    --model models/yolov8s_fp32.onnx \
    --output models/yolov8s_openvino.xml \
    --imgsz 640
# → models/yolov8s_openvino.xml  +  models/yolov8s_openvino.bin
```

`--model` defaults to `models/yolov8s_fp32.onnx` and is resolved at run time. **If the ONNX is
missing but `models/yolov8s.pt` exists, convert exports the ONNX from the `.pt` first** (same
defaults as `python main.py export`, honoring convert's `--imgsz`), then proceeds with the
conversion. If neither file exists, the standard missing-path error is raised (pointing at
ARTIFACTS.md for the git-ignored artifacts).

By default the IR is **FP16** — half the model size; the accuracy effect is
a measured quantity. Measure it against the **ONNX FP32 baseline**
(`models/yolov8s_fp32.onnx` — not the `.pt`: the IR was converted from the ONNX,
so comparing to the ONNX isolates the conversion-plus-precision delta):
`benchmark --validation` (the native COCO evaluator) drives `OpenVINOEngine` for
both `openvino:` and `openvino_int8:` model prefixes, so mAP50 / mAP50-95 for the
FP16 and INT8 IRs are directly comparable against the ONNX FP32 backend. Adding
the FP32, FP16, and INT8 IRs to one benchmark run decomposes the total INT8-vs-FP32
delta into FP16-conversion + NNCF-quantization cost. Pass `--no-fp16` if you
want full FP32 (rarely useful at inference time).

The raw-tensor stats and the detection-level verdict are recorded by the two
consistency runs below (see the provenance note after the commands). The
consistency command accepts the same backend prefixes as benchmark, so the
pair is:

```bash
# FP16 IR, tensor mode — records the raw deltas. EXPECTED to FAIL the
# per-element gate even at the default tolerances: FP16 weight rounding
# should leave the tensor bulk intact but shift a tail of box coordinates
# by px-scale amounts — the same low-impact tail ARCHITECTURE.md §11
# documents for INT8.
# Read its stats, don't read its PASS/FAIL; the verdict is detection mode.
# Measured (2026-10-24, see the provenance note below): mean_diff 0.0100 /
# p95 0.0473 / p99 0.1816 / box max_diff 5.57 (3.53 batch-mean) /
# cosine 0.9999998 — cls channels stay at ~1e-7 mean diff; the entire
# delta is the box-coordinate tail. fail_rate 100% at the per-element
# gate, as expected.
python main.py consistency \
    --model1 models/yolov8s_fp32.onnx \
    --model2 openvino:models/yolov8s_openvino.xml \
    --imgs-input data --mode tensor \
    --report-path results/consistency_openvino_fp16.json

# INT8 IR, detection mode — the FP32↔INT8 convention (IoU/class/score
# gates absorb px-scale box tails), directly comparable to the ORT INT8
# consistency result. Measured (2026-10-24, see the provenance note
# below): 14/16 images pass; mean IoU 0.981, class match 1.000,
# score diff 0.020, count diff 0.25, recall match 0.927, advisory
# cosine 0.9990 — fail_rate 12.5% (the same-order pass band as the ORT
# INT8 path).
python main.py consistency \
    --model1 models/yolov8s_fp32.onnx \
    --model2 openvino_int8:models/yolov8s_openvino_int8.xml \
    --imgs-input data --mode detection \
    --report-path results/consistency_openvino_int8.json
```

(Each run inserts its own `_<TS>` into the stem, so the commands above write
`consistency_openvino_fp16_<TS>.json` / `consistency_openvino_int8_<TS>.json`;
`<TS>` is fixed-width and sortable, so the newest report is selected by name
sort, e.g. `ls results/consistency_openvino_*.json | sort | tail -1`.)

Provenance of the measured figures above (reproducibility rules):

* Date: 2026-10-24.
* Host: Windows 11 + WSL2 Ubuntu, Python 3.12.3, 13th Gen Intel Core
  i5-13420H (12 threads, no AVX-512).
* Packages: openvino 2026.3.0, nncf 3.3.0, onnxruntime 1.26.0,
  torch 2.6.0+cpu.
* Config: `OPENVINO_DEVICE=CPU` (default; actual device CPU, no fallback),
  imgsz 640, batch 1; default `--max-images` budget (20) → 16
  sampler-selected images from the 397-image val split; reports under
  `results/`, located by the newest-name sort rule above.
* Reading the FP16 result: the raw-tensor FAIL is expected and carries no
  verdict — the per-element gate flags px-scale box tails while the cls
  channels sit at ~1e-7 mean diff and cosine ≥ 0.9999998. The INT8
  detection verdict is the release-relevant one: mean IoU 0.981 with
  class match 1.000 and score diff 0.020 sits in the same band as the
  ORT INT8 path's detection run (2026-10-15, 13/16 images, mean IoU
  0.980); the 2 failing images are both recall-gated (recall 0.50 / 0.33,
  count_diff 2, class match 1.0).

**FP16 delta channel split (measured 2026-10-24, one-off analysis; the
analysis record lives under `results/`).**
The per-image `cls_*` stats in the consistency report only cover the cls half of the claim;
so a one-off analysis reran the same pipeline piece for piece — the same sampler selection
(budget 20 → the same 16 images), the same `preprocess_imgs` + `ModelWrapper` forwards — and
split the raw `[1, 16, 8400]` output into box (channels 0:4) and cls (channels 4:) groups,
reporting per-group mean/max/p99 abs diff and cosine and localizing the per-element gate
violations (atol 1e-3 / rtol 1e-2) to the two groups. Result over the 16 images
(2,150,400 elements total):

* **The delta is the box tail, quantitatively.** 646 tolerance violations in total —
  637 (98.6%) in the box channels, 9 in cls. Box: mean abs diff 0.0398, p99 ≤ 0.486,
  max 5.57 (reproduces the recorded tensor-run max), cosine ≥ 0.9999998. Cls:
  mean abs diff 2.2e-7, min cosine 0.99997 — float noise, as claimed.
* **The 9 cls violations are one outlier element** (0.0121 on one image's logit,
  every other image ≤ 2.9e-3) — negligible after the head's sigmoid, and irrelevant
  to detection mode, which is the verdict that counts.
* **w/h channels carry more delta than x/y** (mean abs diff 0.075 / 0.050 vs
  0.017 / 0.017; max 5.57 / 4.15 vs 2.52 / 1.95) — consistent with the DFL decode
  exponentiating widths/heights, which amplifies FP16 rounding.
* Cross-check: box mean 0.0398 over 4 of 16 channels ≈ the recorded run's overall
  `mean_diff` 0.0100 — the two measurements are one consistent dataset.

Implementation notes: `ModelWrapper` in `src/consistency.py` dispatches the
`openvino:`/`openvino_int8:` prefixes to an `OpenVINOEngine` (the prefix is
stripped head-only, the `.xml` is existence-checked, and a non-`.xml` suffix
is rejected at construction — before the lazy engine import).
`validate_consistency` compiles the IR — device from `OPENVINO_DEVICE`,
default CPU — in `ModelWrapper._load_model` at wrapper construction, so a
missing install or an unservable device fails the run before any measurement.
`--device` is not consulted for the OpenVINO side — a `--device cuda` run is
an ORT-CUDA vs OpenVINO cross-device comparison, valid but deliberate. The
report records both the requested and the actual OpenVINO device
(`openvino_devices` — `engine.device` after any CPU fallback). The two
measurements are complementary: `benchmark --validation` reports full-val
aggregate mAP for raw OpenVINO IRs through the native evaluator
(`utils/map_eval`); a consistency run adds the per-image gate detail
(IoU / score / class) that an aggregate mAP does not show.

The converter uses `ov.convert_model()` from the 2023.1+ API. The input shape's **batch dimension
mirrors the ONNX's** (`_resolve_input_shape` reads the ONNX's first input — static `[1,3,...]` if
the ONNX is static-batch, dynamic `[-1,3,...]` if dynamic — mirror, never force). This matters for a
*static* export: the Detect head's DFL Reshape constant is baked to the export batch
(`[1,4,16,8400]` for static-batch-1), so forcing a different batch on the IR crashes mid-graph. The
current `yolov8s_fp32.onnx` is, however, **dynamic-batch** (input is the symbolic dim `batch`; the
DFL Reshape dims are `[-1,4,16,8400]` / `[-1,4,8400]` — batch propagates dynamically, it is *not*
baked to 1), so the IR is `[-1,3,640,640]` and `--batch-size 8` runs as a real 8-image forward (no
sub-looping). Spatial dims are pinned to `imgsz` so OpenVINO picks fully-shaped oneDNN/VNNI kernels.
The engine exposes that pinned size as `OpenVINOEngine.spatial_size` (read from
`.partial_shape[2]/[3]`, `None` when dynamic) and rejects at construction an `imgsz` that disagrees
with it (both sizes named in the error). The engine reads `_model_batch` from the compiled input's
`.partial_shape[0].get_length()` *inside a try* — `.partial_shape[0]` is a `Dimension` whose
`get_length()` returns the int for a static dim and raises for a dynamic one
(`partial_shape` is the one accessor that works on both static and dynamic
IRs); dynamic → `_model_batch=None` → `_effective_batch` honors
`--batch-size`. For a genuinely static-batch IR, `_effective_batch` sub-loops at the baked batch and
`_forward` zero-pads the trailing partial batch then slices the output back — no crash, no dropped
image. For true batched throughput on a *static* export, re-export the ONNX with a dynamic batch
dim.

**FP16 weight compression happens at save time — and the flag is passed explicitly in both
directions.** `compress_to_fp16` is a CLI-only flag for the `ovc`/`mo` command-line tools; the
Python `ov.convert_model()` accepts neither it nor a `data_type` kwarg, and
`nncf.CompressWeightsMode` carries no FP16 member (INT8/INT4/NF4/FP8 only) — all verified on the
pinned openvino 2026.3.0 / nncf 3.3.0. The Python counterpart is
`ov.save_model(model, path, compress_to_fp16=...)`, whose **default is `True`** on 2026.3.0
("Floating point weights are compressed to FP16 by default"). The converter passes the flag
explicitly for both `fp16=True` and `fp16=False`, making `--no-fp16` produce true FP32 weights.
Builds without the kwarg fall back to
the build default and the log reports `actual_fp16=None` (unknown — no precision is asserted); the
convert log always records both `requested_fp16` and `actual_fp16`, so a silent degradation cannot
read as a successful convert.

## Quantize with NNCF (INT8)

```bash
python main.py openvino quantize \
    --model models/yolov8s_fp32.onnx \
    --output models/yolov8s_openvino_int8.xml \
    --imgs-input data \
    --max-cal-samples 300 \
    --subset-size 64
```

Pipeline:

1. ONNX → FP16 IR (intermediate, kept on disk next to the INT8 output)
2. `CalibrationSampler` selects 300 diverse images using the same sqrt-frequency + phash + ResNet-50
   farthest-first strategy as the ORT path
3. NNCF `quantize()` runs on a `--subset-size` subset (default 64 — fast)
4. The Detect head is excluded from INT8 — same *intent and boundary* as the ORT path's
   `HEAD_NAME_PREFIXES=["/model.22/"]`, via a two-layer `IgnoredScope`: a **name pattern**
   (`/model\.22/.*` — the exact Detect subgraph, 135/647 graph nodes on the pinned openvino
   2026.3.0, including the head's Conv branches that an op-type scope cannot express) plus a
   **portable op-type floor** (`Sigmoid`/`Softmax` — YOLOv8s carries exactly one of each, both
   inside the head, so this layer stays head-precise even on builds that rewrote friendly names).
   `validate=False` keeps unmatched entries harmless across IR op-sets; the name layer's resolution
   is checked against the actual graph and logged
   (`Head exclusion: N graph nodes match /model.22/...`), with a loud warning when N=0 so a
   portability fallback is never silent
5. Top-level `fast_bias_correction=True` (the NNCF 3.x API) — adjusts BN/Conv bias after calibration
   to recover accuracy lost to the quantization bias shift

Optional flags:

- `--smooth-quant` — moves activation outliers into weights via scale equivalence. **Gating behavior
  (verified on the pinned nncf 3.3.0):** the NNCF PTQ pipeline adds the SmoothQuant step *only* when
  `model_type=TRANSFORMER` — passing `smooth_quant_alphas` alone has no effect. The flag therefore
  passes `model_type=transformer` explicitly and pins `preset=performance` (TRANSFORMER by default
  flips the preset to `mixed`, confounding any A/B). One residual confounder is disclosed, not
  controlled: TRANSFORMER also auto-disables batchwise statistics. Validate the *net* accuracy
  effect (flag on vs off, against the FP16 baseline, e.g. through `benchmark
  --validation`) before trusting it — an opt-in guard
  test (`test_smooth_quant_is_gated_on_transformer_model_type`) pins the gate itself.
- `--subset-size N` — NNCF runs PTQ over N of the `--max-cal-samples` sampled images (default 64 —
  fast).

## Run Inference

```bash
# CPU
python main.py openvino run \
    --model models/yolov8s_openvino_int8.xml \
    --imgs-input data --device CPU

# Intel iGPU / Arc dGPU
python main.py openvino run \
    --model models/yolov8s_openvino.xml \
    --imgs-input data --device GPU

# Auto-select best device
python main.py openvino run \
    --model models/yolov8s_openvino.xml \
    --imgs-input data --device AUTO

# Heterogeneous (try GPU first, fall back to CPU)
python main.py openvino run \
    --model models/yolov8s_openvino.xml \
    --imgs-input data --device HETERO:GPU,CPU
```

**Device availability is preflighted at engine init.** The engine logs
`OpenVINO devices available: [...]` and resolves a `--device` request that
none of the enumerated devices can serve (compound `MULTI:`/`HETERO:` forms
need every named component present; `AUTO` always passes) to an **explicit
CPU fallback**: a warning names the unavailable request, the device list,
and a driver hint, and the run continues on CPU. The preflight exists because
OpenVINO's own failure for an absent device is an opaque compile-time C++
exception — the run-level failure carries no actionable message. The
fallback is never silent —
`engine.device` always records what actually ran, so a benchmark can't
mistake CPU numbers for iGPU numbers. The request only errors when CPU
itself is unavailable.

> **GPU shows as unavailable under WSL2?** The pip `openvino` package ships
> the GPU plugin, but the plugin enumerates devices through the Level Zero
> driver: install the **Windows-side Intel GPU driver with WSL compute
> support** and verify `/dev/dxg` exists inside WSL (`ls -l /dev/dxg`).
> Without it, `available_devices` is `['CPU']` even on a box with Intel
> integrated graphics; `--device GPU` runs on CPU with a warning meanwhile.

## Benchmark

Add `openvino` to the existing benchmark pipeline:

```bash
python main.py benchmark \
    --model onnx_fp32:models/yolov8s_fp32.onnx \
    --model onnx_int8:models/yolov8s_int8.onnx \
    --model openvino:models/yolov8s_openvino.xml \
    --model openvino_int8:models/yolov8s_openvino_int8.xml \
    --imgs-input data --max-images 32 --warmup 10 --runs 20

# Benchmark OpenVINO on the Intel iGPU (OPENVINO_DEVICE drives the engine's device)
OPENVINO_DEVICE=GPU python main.py benchmark \
    --model openvino:models/yolov8s_openvino.xml \
    --imgs-input data
```

## Measured Performance (Intel Core i5-13420H, YOLOv8s 640×640, batch=1)

> Provenance: recorded 2026-10-24, idle host, Windows 11 + WSL2 Ubuntu,
> Python 3.12.3, 13th Gen Intel Core i5-13420H (12 threads, no AVX-512);
> openvino 2026.3.0, nncf 3.3.0, onnxruntime 1.26.0, torch 2.6.0+cpu;
> imgsz 640, batch 1, 14 images (sampler budget 16), warmup 5, runs 10;
> `OPENVINO_DEVICE=CPU` (no fallback — both OpenVINO rows resolved to
> CPU); the INT8 IR is quantized through `openvino quantize` (its
> head-exclusion scope — name-pattern `/model.22/` + `Sigmoid`/`Softmax`
> floor, defined in `src/openvino_convert.py` — keeps the Detect head out
> of INT8; its weights are written FP16 by `ov.save_model`'s default
> compression); summary CSV under `results/` (the newest
> `benchmark_summary_*.csv` by name sort).
>
> Column semantics: `p50`/`p99` are percentiles of one **full benchmark run**
> (all images, end-to-end: preprocess → forward → NMS), not per-image
> latency; `FPS = images / mean_total_s` (same convention as the README
> benchmark output).

| Backend         | Precision          | p50 (ms) | p99 (ms) | FPS   | Peak Mem (MB) |
|-----------------|--------------------|----------|----------|-------|---------------|
| PyTorch (eager) | FP32               | 2761     | 2889     | 5.09  | 569           |
| ONNX Runtime    | FP32               | 2037     | 2136     | 6.92  | 698           |
| ONNX Runtime    | INT8 QDQ           | 1282     | 1436     | 10.89 | 608           |
| **OpenVINO**    | **FP16 IR**        | 2483     | 2591     | 5.60  | 932           |
| **OpenVINO**    | **INT8 IR (NNCF)** | 1560     | 1653     | 8.98  | 803           |

Reproduce the run: quantize the INT8 IR with `openvino quantize`, then
benchmark on an idle host:

```bash
python main.py openvino quantize \
    --model models/yolov8s_fp32.onnx \
    --output models/yolov8s_openvino_int8.xml \
    --imgs-input data --imgsz 640 --max-cal-samples 300 --subset-size 64

python main.py benchmark \
    --model pytorch:models/yolov8s.pt \
    --model onnx_fp32:models/yolov8s_fp32.onnx \
    --model onnx_int8:models/yolov8s_int8.onnx \
    --model openvino:models/yolov8s_openvino.xml \
    --model openvino_int8:models/yolov8s_openvino_int8.xml \
    --imgs-input data --max-images 16 --warmup 5 --runs 10
```

Reading the table (same recorded run, 2026-10-24):

* **INT8 across both runtimes:** ORT INT8 QDQ (10.89 FPS) vs OpenVINO INT8
  IR (8.98 FPS) — on this silicon the ORT QDQ path leads, and both INT8
  forms beat their FP32/FP16 counterparts (ORT FP32 6.92, OpenVINO FP16
  5.60 FPS). The hypothesis that OpenVINO's VNNI INT8 kernels dominate ORT
  QDQ on a no-AVX-512 part is not supported by this run.
* **FP16 IR:** slower than ONNX FP32 (5.60 vs 6.92 FPS) — FP16 weight
  compression on a CPU-only path buys footprint, not speed, here.
* **Footprint:** peak memory orders OpenVINO FP16 (932 MB) > OpenVINO INT8
  (803 MB) > ORT FP32 (698 MB) > ONNX INT8 (608 MB) ≈ PyTorch (569 MB);
  the OpenVINO runtime carries a higher resident baseline on this host.
* All five rows come from one process; per-backend RSS increase and the
  full percentile set are in the CSV.

Not measurable in the validated set:

| Platform | Status |
|---|---|
| Intel iGPU / Arc dGPU | not measurable — no Intel GPU in the validated hardware set |
| NVIDIA T4 / Jetson | n/a for OpenVINO (Intel-only runtime); ONNX Runtime's CUDA execution provider covers these |

## Python API

```python
from src import OpenVINOEngine, OpenVINOAsyncEngine

# Synchronous — same API as YOLOv8Engine
engine = OpenVINOEngine(
    model_path="models/yolov8s_openvino_int8.xml",
    device="CPU",
    imgsz=640,
    num_streams="AUTO",
)
detections = engine.infer(imgs_input="data", conf=0.3, iou=0.45)

# Async — pipelined for video streams
async_engine = OpenVINOAsyncEngine(
    model_path="models/yolov8s_openvino_int8.xml",
    device="CPU",
    n_requests=4,   # 4 inference requests in flight
)
from src import preprocess_frames
for frame_id, frame in enumerate(video_frames):
    data = preprocess_frames([frame], imgsz=640)
    async_engine.start_batch(frame_id, data["images"].cpu().numpy())
# Results return in COMPLETION order — re-key by frame_id to restore order
results = dict(async_engine.wait_and_get())
```

Async contract notes: `start_batch` routes through the same
`_pad_to_static_batch` helper as the sync `_forward`, so a static-batch IR
never receives a partial batch on either path (outputs are sliced back to the
real image count in the callback). Callbacks store a **copy** of each output —
`AsyncInferQueue` recycles its request buffers on subsequent rounds, so a view
would be silently overwritten.

## Design Decisions

The main deliberate choices on this path, each expanded in its own section:

* **ONNX FP32 as the interchange first.** The model is exported to ONNX once
  and converted from there via `ov.convert_model`, so the OpenVINO path sits
  on the same upstream artifact as the ORT path.
* **Detect-head exclusion mirrors the ORT boundary.** The head is excluded
  from INT8 quantization through a two-layer `IgnoredScope` (name pattern
  `/model.22/` + a `Sigmoid`/`Softmax` op-type floor); its weights are
  written FP16 by `ov.save_model`'s default weight compression. The resolved
  node count is logged and N=0 raises a loud warning, so a name-rewriting
  build cannot degrade the exclusion silently. See
  Quantize with NNCF, step 4.
* **SmoothQuant gating made explicit.** NNCF adds the SmoothQuant step only
  under `model_type=TRANSFORMER` — passing the alphas alone has no effect —
  so the flag sets the model type explicitly and pins the preset to
  keep A/B runs attributable. See Optional flags.
* **Batch-dimension contract.** The converter mirrors the ONNX's batch dim
  (the Detect head's DFL reshape is baked to the
  export batch on static exports); the engine reads the batch back off the
  compiled input's partial shape, and the sync and async paths share one
  `_pad_to_static_batch` helper, so no partial batch ever reaches a
  static-batch graph and no image is dropped. See Convert ONNX → OpenVINO IR.
* **Compile at construction.** `OpenVINOEngine`'s constructor calls
  `compile_model` once; every `infer()` reuses the compiled model, and
  `_model_batch` / `spatial_size` are read from the compiled input's partial
  shape — properties of the compiled model. Construction also runs the device
  preflight, so an unservable device fails at engine creation, before any
  measurement. See Python API.
* **Figures come from the recorded run.** The table reports what the bench
  says, negative results included: on the validated i5-13420H (no AVX-512),
  the hypothesis to test is that any OpenVINO advantage comes from VNNI INT8
  kernels, with the FP16 IR contributing little — the recorded run fills the
  table and the reading under Measured Performance.
