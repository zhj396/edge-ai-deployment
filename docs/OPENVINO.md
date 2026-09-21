# OpenVINO Deployment

Intel's OpenVINO toolkit is the de-facto runtime for Intel CPUs, Intel integrated GPUs, Intel Arc GPUs, and Intel Movidius VPUs. This toolchain supports it as a fourth inference backend alongside PyTorch, ONNX FP32, and ONNX INT8.

## Why OpenVINO for Edge AI

| Hardware | Best backend | Why |
|---|---|---|
| Intel Xeon / Core CPU | **OpenVINO** | oneDNN + AVX-512 / VNNI tuning; on this repo's validated i5-13420H: `_TBD_` pending measurement (see Measured Performance) |
| Intel iGPU / Arc dGPU | **OpenVINO** | OpenVINO GPU plugin, free on every Intel box |
| Movidius VPU (USB stick / RPi AI Kit) | **OpenVINO** | *Only* runtime that supports Myriad X |
| NVIDIA Jetson / dGPU | ONNX Runtime (CUDA) / TensorRT | OpenVINO is Intel-only |
| ARM Cortex-A | ONNX Runtime | OpenVINO ARM support exists but limited |

For DACH industrial customers (Bosch, Siemens, Continental, ABB, …) OpenVINO is essentially table stakes — most production lines run on Intel hardware.

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

`--model` defaults to `models/yolov8s_fp32.onnx` and is resolved at run time. **If the ONNX is missing but `models/yolov8s.pt` exists, convert exports the ONNX from the `.pt` first** (same defaults as `python main.py export`, honoring convert's `--imgsz`), then proceeds with the conversion. If neither file exists, the standard missing-path error is raised (pointing at ARTIFACTS.md for the git-ignored artifacts).

By default the IR is **FP16** — half the model size; accuracy impact on YOLOv8s is `_TBD_` pending a consistency run (`python main.py consistency` against the ONNX FP32 baseline). Pass `--no-fp16` if you want full FP32 (rarely useful at inference time).

The converter uses `ov.convert_model()` from the 2023.1+ API. The input shape's **batch dimension mirrors the ONNX's** (`_resolve_input_shape` reads the ONNX's first input — static `[1,3,...]` if the ONNX is static-batch, dynamic `[-1,3,...]` if dynamic — mirror, never force). This matters for a *static* export: the Detect head's DFL Reshape constant is baked to the export batch (`[1,4,16,8400]` for static-batch-1), so forcing a different batch on the IR crashes mid-graph. The current `yolov8s_fp32.onnx` is, however, **dynamic-batch** (input is the symbolic dim `batch`; the DFL Reshape dims are `[-1,4,16,8400]` / `[-1,4,8400]` — batch propagates dynamically, it is *not* baked to 1), so the IR is `[-1,3,640,640]` and `--batch-size 8` runs as a real 8-image forward (no sub-looping). Spatial dims are pinned to `imgsz` so OpenVINO picks fully-shaped oneDNN/VNNI kernels. The engine reads `_model_batch` from the compiled input's `.partial_shape[0].get_length()` *inside a try* — `.partial_shape[0]` is a `Dimension` whose `get_length()` returns the int for a static dim and raises for a dynamic one (the plain `.shape` accessor is unusable here: it AttributeErrors on a static IR and raises "to_shape was called on a dynamic shape" on a dynamic IR); dynamic → `_model_batch=None` → `_effective_batch` honors `--batch-size`. For a genuinely static-batch IR, `_effective_batch` sub-loops at the baked batch and `_forward` zero-pads the trailing partial batch then slices the output back — no crash, no dropped image. For true batched throughput on a *static* export, re-export the ONNX with a dynamic batch dim.

**FP16 weight compression happens at save time — and the flag is passed explicitly in both directions.** `compress_to_fp16` is a CLI-only flag for `ovc`/`mo`; the Python `ov.convert_model()` accepts neither it nor a `data_type` kwarg, and `nncf.CompressWeightsMode` carries no FP16 member (INT8/INT4/NF4/FP8 only) — all verified on the pinned openvino 2026.3.0 / nncf 3.3.0. The Python counterpart is `ov.save_model(model, path, compress_to_fp16=...)`, whose **default is `True`** on 2026.3.0 ("Floating point weights are compressed to FP16 by default") — so a plain save would silently write FP16 weights even for `--no-fp16`. The converter therefore passes the flag explicitly for both `fp16=True` and `fp16=False`, making `--no-fp16` produce true FP32 weights. Builds without the kwarg fall back to the build default and the log reports `actual_fp16=None` (unknown) rather than guessing; the convert log always records both `requested_fp16` and `actual_fp16`, so a silent degradation can never read as a successful convert.

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
2. `CalibrationSampler` selects 300 diverse images using the same sqrt-frequency + phash + ResNet-50 farthest-first strategy as the ORT path
3. NNCF `quantize()` runs on a `--subset-size` subset (default 64 — fast)
4. The Detect head is excluded from INT8 — same *intent and boundary* as the ORT path's `HEAD_NAME_PREFIXES=["/model.22/"]`, via a two-layer `IgnoredScope`: a **name pattern** (`/model\.22/.*` — the exact Detect subgraph, 135/504 graph nodes on the pinned openvino 2026.3.0, including the head's Conv branches that an op-type scope cannot express) plus a **portable op-type floor** (`Sigmoid`/`Softmax` — YOLOv8s carries exactly one of each, both inside the head, so this layer stays head-precise even on builds that rewrote friendly names). `validate=False` keeps unmatched entries harmless across IR op-sets; the name layer's resolution is checked against the actual graph and logged (`Head exclusion: N graph nodes match /model.22/...`), with a loud warning when N=0 so a portability fallback is never silent
5. Top-level `fast_bias_correction=True` (the NNCF 3.x API) — adjusts BN/Conv bias after calibration to recover accuracy lost to the quantization bias shift

Optional flags:

- `--smooth-quant` — moves activation outliers into weights via scale equivalence. **Gating gotcha (root-caused on the pinned nncf 3.3.0):** the NNCF PTQ pipeline adds the SmoothQuant step *only* when `model_type=TRANSFORMER` — passing `smooth_quant_alphas` alone is a silent no-op. The flag therefore passes `model_type=transformer` explicitly and pins `preset=performance` (TRANSFORMER would otherwise flip the preset to `mixed`, confounding any A/B). One residual confounder is disclosed rather than controlled: TRANSFORMER also auto-disables batchwise statistics. Validate the *net* effect with a consistency run (flag on vs off, against the FP16 baseline) before trusting it — an opt-in guard test (`test_smooth_quant_is_gated_on_transformer_model_type`) pins the gate itself.
- `--subset-size N` — NNCF runs PTQ over N of the `--max-cal-samples` sampled images (default 64 — fast).

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
and a driver hint, and the run continues on CPU instead of failing with
OpenVINO's opaque compile-time C++ exception. The fallback is never silent —
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

# Benchmark OpenVINO on Intel iGPU instead of CPU
OPENVINO_DEVICE=GPU python main.py benchmark \
    --model openvino:models/yolov8s_openvino.xml \
    --imgs-input data
```

## Measured Performance (Intel Core i5-13420H, YOLOv8s 640×640, batch=1)

> ⚠️ **All values below are `_TBD_` placeholders, not estimates.** Fill them
> only from a benchmark run on an **idle host**, with the INT8 IR quantized
> through `openvino quantize` (its head-exclusion scope — name-pattern
> `/model.22/` + `Sigmoid`/`Softmax` floor, defined in `src/openvino_convert.py`
> — keeps the Detect head FP32). Measurements taken while the host runs
> other load understate absolute performance and are not quotable.
>
> Column semantics: `p50`/`p99` are percentiles of one **full benchmark run**
> (all images, end-to-end: preprocess → forward → NMS), not per-image
> latency; `FPS = images / mean_total_s` (same convention as the README
> benchmark output).

| Backend | Precision | p50 (ms) | p99 (ms) | FPS | Peak Mem (MB) |
|---|---|---|---|---|---|
| PyTorch (eager) | FP32 | `_TBD_` | `_TBD_` | `_TBD_` | `_TBD_` |
| ONNX Runtime | FP32 | `_TBD_` | `_TBD_` | `_TBD_` | `_TBD_` |
| ONNX Runtime | INT8 QDQ | `_TBD_` | `_TBD_` | `_TBD_` | `_TBD_` |
| **OpenVINO** | **FP16 IR** | `_TBD_` | `_TBD_` | `_TBD_` | `_TBD_` |
| **OpenVINO** | **INT8 IR (NNCF)** | `_TBD_` | `_TBD_` | `_TBD_` | `_TBD_` |

Fill the table by quantizing the INT8 IR with `openvino quantize`, then
benchmarking on an idle host:

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

When recording the filled table, note per the reproducibility rules: date,
host state (idle), hardware, OS/runtime, package versions, imgsz / batch /
image count / warmup / runs, and the code state (commit).

Reading the filled table: compare OpenVINO INT8 vs ORT INT8 and vs ORT
FP32, and the FP16 IR vs ORT FP32. This CPU (i5-13420H) has no AVX-512;
its fast path is VNNI INT8 kernels, so any OpenVINO advantage on this
silicon comes from the INT8 IR rather than the FP16 IR. Vendor headline
multipliers ("1.5–2× over ORT") are marketing claims, not measurements —
this table reports what this repo's bench measures, negative results
included. OpenVINO's compile-time memory planning also shows up in peak
RSS.

Not measurable in the validated set (recorded as `_TBD_`, not estimated):

| Platform | Status |
|---|---|
| Intel iGPU / Arc dGPU | `_TBD_` — no Intel GPU in the validated hardware set |
| NVIDIA T4 / Jetson | n/a for OpenVINO (Intel-only runtime); ORT CUDA / TensorRT cover these |

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

## Architecture Notes

OpenVINO's two-stage pipeline (compile → infer) matches the typical "compile once, run many" pattern in production. `compile_model()` does the heavy lifting:

* Graph-level optimization (Conv-BN folding, activation fusion)
* Platform-specific kernel selection (oneDNN, MKL-DNN, AVX-512, VNNI)
* Memory planning for static shapes

After compilation, `compiled([input])` is a thin wrapper — the actual kernel call is in C++ land.

## Design Decisions

The main deliberate choices on this path, each expanded in its own section:

* **ONNX FP32 as the interchange first.** The model is exported to ONNX once
  and converted from there via `ov.convert_model`, so the OpenVINO path sits
  on the same upstream artifact as the ORT path rather than a second export
  route.
* **Detect-head exclusion mirrors the ORT boundary.** The head stays FP32
  through a two-layer `IgnoredScope` (name pattern `/model.22/` + a
  `Sigmoid`/`Softmax` op-type floor); the resolved node count is logged so a
  version that rewrites names fails loudly instead of silently. See
  Quantize with NNCF, step 4.
* **SmoothQuant gating made explicit.** NNCF adds the SmoothQuant step only
  under `model_type=TRANSFORMER` — passing the alphas alone is a silent
  no-op — so the flag sets the model type explicitly and pins the preset to
  keep A/B runs attributable. See Optional flags.
* **Batch-dimension contract.** The converter mirrors the ONNX's batch dim
  instead of forcing one (the Detect head's DFL reshape is baked to the
  export batch on static exports); the engine reads the batch back off the
  compiled input's partial shape, and the sync and async paths share one
  `_pad_to_static_batch` helper, so no partial batch ever reaches a
  static-batch graph and no image is dropped. See Convert ONNX → OpenVINO IR.
* **Measured, not assumed.** Performance figures stay `_TBD_` until the INT8
  IR is quantized and benched on an idle host; the table reports what the
  bench says, negative results included. On the validated i5-13420H
  (no AVX-512), any OpenVINO advantage is expected to come from VNNI
  INT8 kernels rather than FP16 weights — which is itself a hypothesis
  the table has to confirm or refute. See Measured Performance.
