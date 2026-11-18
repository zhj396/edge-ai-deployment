# QNN Deployment (Qualcomm HTP, on-device)

The Qualcomm AI Engine (QNN) executes inference on the AR1+ smart-glasses SoC's
Hexagon HTP (v73) through a *host-driven device workflow*: QAIRT host tools convert
the FP32 ONNX into a PTQ context binary, the context + inputs travel over adb, and
`qnn-net-run` executes on the device. This backend is unlike every other backend in
this toolchain — the accelerator is not on the benchmark host — and the two graph
contracts it introduces (NHWC input layout, split box/cls outputs) are documented
below alongside the conversion pipeline.

## Why QNN for Edge AI

| Hardware | Best backend | Why |
|---|---|---|
| Qualcomm SoC (Hexagon HTP) | **QNN** | the NPU path: HTP v73 executes the whole u8/fp16 graph on the DSP; no other backend here reaches a real NPU |
| NVIDIA GPU | TensorRT / ONNX Runtime (CUDA) | different vendor |
| Intel CPU / iGPU | OpenVINO | different vendor |
| Generic CPU | ONNX Runtime | portable default |

## Pinned environment

The validated environment (recorded in the runs below): QAIRT SDK **2.42.0.251225**
(the zip layout; the SDK's `libQnnHtp.so` / `libQnnHtpV73Stub.so` /
`libQnnHtpV73Skel.so` are sha256-identical to the device's `/system_ext/lib64/`
copies — the SDK version must match the device BSP exactly), Python 3.10 for the
host converter (`~/env_qnn`: `numpy==1.26.4`, `protobuf==3.20.3`, `onnx==1.16.2` —
numpy 2.x / protobuf 7.x break the converter's Python↔C bridge
`libPyIrGraph.so`), Android NDK r27c (only `qnn-model-lib-generator` needs it),
and adb (platform-tools). Device: AR1+ (SAR2130P / Aliso SG2, soc_id 740), Android
14 userdebug, QNN v2.42.0 userspace in `/system_ext/lib64`, CDSP up.

Install the SDK by extracting the QAIRT zip and pointing `QNN_SDK_ROOT` at it:

```bash
export QNN_SDK_ROOT=~/qairt/qairt/2.42.0.251225   # bin/x86_64-linux-clang + lib/aarch64-android
export QNN_ADB=/path/to/adb                        # or put adb on PATH
export QNN_DEVICE_SERIAL=d7aea176                  # optional pin when several devices are attached
```

The SDK is a zip download (Qualcomm Software Center / package manager), not a pip
package — there is no `requirements-qnn.txt`. The repo-side code
(`src/qnn_engine.py`) needs only `numpy` + the adb client; the host *converter*
environment needs the Python 3.10 pin set above.

## Three conversion stages (host) → context binary

The canonical route is **PTQ** (post-training quantization via `--input_list`).
Two alternatives are blocked by the QAIRT 2.42 host compiler and are documented
negative results (see "Host-compiler blockers" below): importing the ORT QDQ
INT8 ONNX directly, and whole-FP32 conversion.

```bash
# Stage 1 — ONNX → QNN model (PTQ: the quantizer calibrates from the input list).
# The shared sampler picks the images; preprocessing runs through the shared
# preprocess_imgs and the raws are written NHWC (see "NHWC input layout").
qnn-onnx-converter --input_network models/yolov8s_fp32.onnx \
  --input_dim images 1,3,640,640 \
  --input_list calib_raw_nhwc/input_list.txt \
  --out_node /model.22/Mul_5_output_0 \
  --out_node /model.22/Sigmoid_output_0 \
  --output yolov8s_w8a8
# W8A16 (head-protection configuration): add --act_bitwidth 16 (weights stay u8).

# Stage 2 — model lib (aarch64-android for the device; the host ctx generator
# loads the x86_64-linux-clang build).
qnn-model-lib-generator -c yolov8s_w8a8.cpp -b yolov8s_w8a8.bin \
  -o . -t aarch64-android x86_64-linux-clang

# Stage 3 — HTP context binary (backend extensions carry soc_id / dsp_arch /
# vtcm_mb; the device's VTCM is 2 MB — a bigger plan fails to load on-device).
qnn-context-binary-generator \
  --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
  --model x86_64-linux-clang/libyolov8s_w8a8.so \
  --binary_file yolov8s_w8a8.serialized \
  --output_dir . \
  --config_file htp_backend_conf.json
```

The backend-extensions config (`htp_backend_conf.json`) points at the NetRun
extensions library and a perf config; the perf config pins the device and graph
plan (the key is `"device"` — an object — not the `"devices"` array syntax the
docs show, which fails to parse on 2.42):

```json
{"device": {"soc_id": 740, "dsp_arch": "v73", "pd_session": "unsigned"},
 "graphs": [{"vtcm_mb": 2, "graph_names": ["yolov8s_w8a8"]}]}
```

For on-device burst-mode runs, `qnn-net-run --config_file` takes the same
backend-extensions JSON with `perf_mode: "burst"` inside the `device` block.

### out_node targets — verify against the graph, not the docs

`--out_node` names become the context binary's outputs; a name that does not
exist in the graph only produces a WARNING, and the converter then **prunes the
unreferenced branch silently** — the file sizes barely change and the only
observable is inside `<output>_net.json` (the box path disappears). The box
branch's final op is the dist2bbox `Mul`; on the current opset-17 export it is
`/model.22/Mul_5_output_0` (`Concat_32 → output0` merges `Mul_5_output_0` +
`Sigmoid_output_0`). Always confirm the targets against the ONNX tail nodes and
audit `_net.json` for both paths after conversion.

## NHWC input layout

QNN's native tensor layout is NHWC. `qnn-net-run` reads `--input_list` raws
**in graph layout** (`[1, 640, 640, 3]`) with no transposition: NCHW-ordered
bytes get pixel-interleaved and the graph computes "a correct forward of the
wrong input" — the outputs look statistically sane (bounded ranges, healthy
histograms) but correlate ~0 with the reference. The minimal-graph proof:
a single-Conv FP32 context fed NCHW bytes diverges (`max|d| ≈ 3.0`,
corr 0.03); the same context fed NHWC bytes is **bit-exact**
(`max|d| = 0`, corr 1.00000000). `src/qnn_engine.py::nchw_batch_to_nhwc_raws`
writes the NHWC byte order for every staged input.

## Split box / cls outputs

A merged output tensor carries ONE per-tensor quantization encoding. Dominated
by the box range (0–640), the cls scores (sigmoid, ≤ ~0.12 on the validated
val set) fall below 1 LSB and dequantize to exactly 0.0 — the whole cls branch
reads as zero. The conversion therefore names two outputs (`--out_node` above;
box `(1, 4, 8400)` + cls `(1, nc, 8400)`), each with its own encoding: the
W8A8 cls scale lands at 0.00348 (independent of the box scale 2.51), and
W8A16 lifts cls to a 16-bit encoding (scale 1.5e-5, full 0–1 range).
`src/qnn_engine.py::split_outputs_to_4nc` re-assembles the device outputs
(NHWC planes) into the channel-first `[bs, 4+nc, N]` layout that
`post_process` and `compare_tensors` consume.

## Head-exclusion posture on this backend

The ORT QDQ path keeps the `/model.22/` Detect head FP32 by QDQ omission
(`nodes_to_exclude`); the QNN PTQ route quantizes the whole net (head included)
— the head-FP32 protection does not carry over. The compensation is the
**output split** (cls gets its own encoding, killing the 1-LSB collapse) plus,
for tighter cls fidelity, **W8A16** (`--act_bitwidth 16`: weights u8, activations
fp16). Importing the ORT QDQ artifact directly would carry the head exclusion
1:1, but that route is blocked by the host compiler (below).

### Host-compiler blockers (QAIRT 2.42, documented negative results)

* **QDQ-import** (feeding the ORT QDQ INT8 ONNX to the converter) and
  **whole-FP32** conversion fail host compile: `q::flat_from_vtcm` at
  `_model.10/Resize` (the float/mixed-tensor Resize lowering).
* Replacing Resize with a bit-exact ConvTranspose rewrite (verified
  max|d|=0 vs ORT on the same inputs) hits a second host-compiler bug:
  `q::ForceFormat_Crouton` at the ConvTranspose node, in any precision.
* The targeted `quantization_overrides` route (u8 backbone encodings + float
  head fallback, name-mapped from `_net.json`) builds the intended IR
  (verified in `net.json`: backbone u8 + head float) and dies in the same
  host compile — the override machinery itself works and is reusable when a
  fixed QAIRT lands.

## Device execution

```bash
adb push device_pkg /data/local/tmp/qnn_pkg     # ctx + libs + qnn-net-run + calib raws
adb shell 'export LD_LIBRARY_PATH=/data/local/tmp/qnn_pkg:/system_ext/lib64
export ADSP_LIBRARY_PATH="/data/local/tmp/qnn_pkg;/system_ext/lib64;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp"
cd /data/local/tmp/qnn_pkg
./qnn-net-run --retrieve_context yolov8s_w8a16.serialized.bin \
  --input_list input_list_nhwc.txt --output_dir out \
  --backend libQnnHtp.so'
```

The device BSP ships no `libQnnHtpPrepare.so` — on-device graph compilation is
unsupported, so the host-side context binary is the only viable flow
(`--retrieve_context`). The device needs `qnn-net-run` +
`libQnnHtp.so`/`libQnnSystem.so`/`libQnnHtpNetRunExtensions.so`
(aarch64-android, SDK-matched) staged next to the context; the V73
Stub/Skel/`.cat` are already in `/system_ext/lib64`.

## Repo integration

* `consistency --model2 qnn:models/yolov8s_w8a16.serialized.bin` — the `qnn:`
  prefix lazily builds `QNNEngineRunner` (host toolchain + adb probed on the
  first forward, the same contract as `ort_cpp:` / `tensorrt:` / `trt_cpp:`).
  Tolerance auto-selection classifies the spec's stem (an INT8/W8A16-named
  context routes to the loose pair).
* `benchmark --model qnn_w8a8:models/yolov8s_w8a8.serialized.bin
  --model qnn_w8a16:models/yolov8s_w8a16.serialized.bin` — these are the
  **first cross-device rows**: warmup runs the real path once (the first
  invocation pays the one-time CDSP/HTP power-up), then the timed loop covers
  stage → adb transfer → device execute → host NMS per run. The CSV `Device`
  column names the target SoC; the per-millisecond numbers are a cross-device
  protocol and are not comparable with the same-host rows. The backend name
  states the quantization configuration (`qnn_w8a8` / `qnn_w8a16` — no
  `qnn_int8` alias, the name is the configuration, not a QDQ artifact).
  mAP `--validation` skips these rows (no `infer()` engine surface for the
  native evaluator; the precision signal is the consistency harness).
* `src/qnn_engine.py` registers `QNNEngineRunner` + `qnn_available` through
  `src.__init__._LAZY` (the optional-dep ImportError-with-hint contract).

## Power envelope: W8A16 × burst resets the board (recorded negative result)

The perf-mode sweep is not free on this board: running the W8A16 context
(2× activation traffic vs W8A8) under `perf_mode: "burst"` (max CDSP clock)
hard-resets the device — reproduced three times across builds and power
setups (USB-powered, externally powered, reflashed): adb drops at context
load, reconnect shows uptime < 130 s with `bootreason=reboot`, empty pstore,
no vreg/thermal kernel traces (the reset completes before the kernel can
log — a PMIC-level power drop). W8A8 × burst and W8A16 × default both run
stable; the trigger is the combination. The burst datapoint below therefore
covers W8A8 only; the W8A16 burst cell stays unmeasured by design (re-running
it would trade board stability for one table cell).

## Measured performance (recorded runs)

> Recorded on the AR1+ SAR2130P (Aliso SG2, HTP v73, VTCM 2 MB), QNN
> v2.42.0.251225 device userspace = host SDK (sha256-matched libs), Android 14
> userdebug build `UKQ1.260430.001/890` + `/1177` (both recorded; the ctx and
> numbers are build-independent). Static batch 1, 1×3×640×640, 8 calibration
> images (seed-42 sampler, NHWC raws). Latency from
> `qnn-net-run --profiling_level detailed` parsed with `qnn-profile-viewer`
> (`Accelerator (execute) time`); the average includes first-inference
> VTCM/HVX power-up (the kernel-cycle column is stable ±0.1%).

| configuration | accelerator execute (avg) | cycles | NetRun e2e (avg) |
|---|---|---|---|
| W8A8, default perf mode | 34.6 ms | 19.63M | 41.0 ms |
| W8A8, burst perf mode | **17.3 ms** | — | 20.1 ms |
| W8A16, default perf mode | 64.5 ms | — | 71.8 ms |
| W8A16, burst perf mode | unmeasured — board resets (see above) | — | — |

The W8A8 : W8A16 cycle ratio (1.86×) reflects the doubled activation traffic
(spill 17.4 vs 11.0 MB in the graph plans); burst halves the W8A8 wall time
by raising the CDSP clock (the kernel cycles are unchanged — same graph).

## Consistency (recorded runs)

> vs the canonical ORT FP32 ONNX on the formal gate image set: the 20-image
> consistency request resolves to **16 images** (the seed-42 sampler's
> farthest-first trim on the val-pool quota — the same sampler behavior the
> ORT-side runs record); `compare_tensors` at the deployed conf 0.25, bs=1
> 3-D tensors (the shape the nc-split activates on). Raw `allclose` 100% fail
> on both configs is the structural reduced-precision reading (the dual
> verdict contract — `fail_rate` and `detection_fail_rate` are always quoted
> as a pair).
>
> **Output layout note (measured on both sets):** the graph *input* contract
> is NHWC (`qnn-net-run` reads raws in `[1,640,640,3]`), while the *output*
> raws score correctly read channel-first (`(4, 8400)` / `(12, 8400)` planes
> — NCHW-planes interpretation corr 0.9988 vs NHWC-transpose −0.111 on the
> same bytes). `src/qnn_engine.py` stages inputs NHWC and loads outputs
> channel-first. Both the earlier probe set and this formal gate set produce
> the same readings under this interpretation.

| pair | tensor `fail_rate` | conf-cliff `detection_fail_rate` | signature |
|---|---|---|---|
| FP32-ONNX ↔ qnn W8A8 (auto-loose) | 100% | 12.5% (2/16) | box corr 0.9985, cls corr 0.9857; cls_cos 0.876–0.9995; flips img8 46→37 (d9) + img12 12→2 (d10) |
| FP32-ONNX ↔ qnn W8A16 (auto-loose) | 100% | 6.25% (1/16) | box corr 0.9996, cls corr 0.9910; cls_cos 0.957–0.9995; flip img12 12→2 (d10) |

The formal gate reads at a larger image count than the earlier probe set:
at 16 images both
configurations cross the 5/img cliff-flip gate on isolated hard images (the
same image — img12 — flips under both, W8A8 additionally on img8), while box
correlation holds at 0.9985/0.9996 and the W8A16 config stays cleaner on both
the cls band and the flip count. The whole-net-PTQ cls collapse seen on the
single-output route (cls scale 2.534 dominated by the box range → cls < 1 LSB
→ all-zero) does not reproduce here: the output split gives cls its own
encoding, and cls correlation stays ≥ 0.9857 on the formal set.
