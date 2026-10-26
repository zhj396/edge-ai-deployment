# ONNX Runtime C++ backend (`cpp/`)

A standalone C++
YOLOv8s inference driver built on the ONNX Runtime C++ API. `cpp/` is a standalone
tree beside the layered Python package — it shares no code with `cli/`+`src/`+`utils/`
and consumes the exported ONNX artifact directly.

## Layout

```
cpp/
├── CMakeLists.txt                 # top: OpenCV, FetchContent (cxxopts/yaml-cpp), ORT SDK, add_subdirectory
├── third_party/
│   ├── onnxruntime-linux-x64-1.20.0.tgz   # committed ONNX Runtime SDK tarball
│   ├── setup-onnxruntime.sh       # standalone extract helper
│   └── onnxruntime/               # extracted ONNX Runtime SDK (headers + lib + cmake; gitignored)
├── common/                        # shared C++ core — mirrors src/preprocess.py + src/postprocess.py
│   ├── CMakeLists.txt             # ort_core_common static lib (OpenCV only — NO ORT dep)
│   ├── include/
│   │   ├── types.h                # Detection / PreprocessResult / ProfileResult
│   │   ├── letterbox.h            # letterboxInto() — cv::Mat only, backend-agnostic
│   │   ├── nms_decode.h           # parseOutputShape() + decode_nms() — const float* only
│   │   ├── logger.h  timer.h  yaml_classes.h  config.h  draw_utils.h
│   └── src/
│       ├── letterbox.cpp  nms_decode.cpp  yaml_classes.cpp  draw_utils.cpp
└── onnxruntime/                   # the ORT C++ backend
    ├── CMakeLists.txt             # ort_cpp exe: links ort_core_common + ORT SDK + cxxopts/yaml
    ├── include/
    │   ├── preprocessor.h         # ORT wrapper: common::letterboxInto → blobFromImage → Ort::Value
    │   ├── postprocessor.h        # ORT wrapper: Ort::Value → float*/shape → common::decode_nms
    │   ├── yolov8.h  cli.h  benchmark.h
    └── src/
        └── preprocessor.cpp  postprocessor.cpp  yolov8.cpp  cli.cpp  main.cpp  benchmark.cpp
```

**Why the split.** `cpp/common/` has **no ONNX Runtime dependency** — it links only OpenCV
(+ the yaml parser). The ORT-specific code (session, `Ort::Value` tensor wrapping) lives in
`cpp/onnxruntime/`. A backend links `ort_core_common` (the documented
shared-core contract) and hands the core a raw `float*` + shape, keeping the
shared preprocess/postprocess core single-sourced in C++.

**SDK path.** The ONNX Runtime SDK lives at `cpp/third_party/onnxruntime/` — *not*
`cpp/onnxruntime/`. The latter path is reserved for the backend *source*. `ONNXRUNTIME_DIR`
env var overrides the SDK location. Note the C++ tree vendors ORT **1.20.0** while the
Python path pins `onnxruntime==1.26.0` (`requirements-cpu.txt`) — the runtimes are
independent installs.

## Numerical consistency vs the Python path

The C++ core is written to be numerically consistent with the Python preprocess/postprocess
(within tolerance — implementation notes below):

- **Letterbox.** `common/letterbox.cpp::letterboxInto` mirrors `src/preprocess.py::letterbox`.
  Resized dimensions are **rounded** (`cv::saturate_cast<int>` ≡ `cvRound`, matching Python's
  `int(round(shape * r))`) so both paths letterbox to identical sizes for non-integer scales.
  The gray fill (114,114,114), `INTER_LINEAR` resize, and centered-ROI copy all match.
  Two boundary cases differ: Python skips `cv2.resize` when the input already matches the
  target size (the C++ path always calls resize — an identity `INTER_LINEAR` resize for
  equal sizes), and Python's `_preprocess_bgr` converts grayscale/BGRA inputs to BGR while
  `letterboxInto` assumes a 3-channel `CV_8UC3` input. Neither case arises for the
  validated 640 COCO-subset configuration.
- **Scale-back + clamp.** `common/nms_decode.cpp::decode_nms` inverts the letterbox with
  `(x - pad) / scale` (single-side pad) — matching `ultralytics.utils.ops.scale_boxes` called
  as `scale_boxes(ratio_pad=((r,r),(pad_x,pad_y)))` in `src/postprocess.py::post_process` —
  then clamps to `[0, w-1] / [0, h-1]`, matching the explicit clamp in `post_process`.
- **/255.** OpenCV applies normalization as multiply-by-`(1/255)` and torch as float32
  division; results can differ by 1 ulp on some pixel values.
- **Int-pixel boxes.** The final box conversion truncates (`static_cast<int>`) where Python
  rounds; drawn boxes can differ by 1px for the same detection (visualization only).
- **NMS — documented divergence, NOT bit-identical.** The C++ path uses
  `cv::dnn::NMSBoxes`, which approximates `ultralytics.utils.nms.non_max_suppression` but is
  not bit-identical (different IoU tie-breaking; ultralytics re-sorts and caps `max_det`).
  A `max_det=300` cap is applied for behavioral parity. This is acceptable here because
  the C++ NMS never enters the cross-backend consistency harness: in tensor mode the
  harness compares raw forward tensors directly, and in detection mode both sides are
  decoded through the same `src/postprocess.py` ultralytics NMS — the C++ `decode_nms`
  only affects the standalone app's drawn boxes, not a consistency input.
  Reimplementing ultralytics NMS in C++ is explicitly out of scope.

## Build

Requires CMake ≥ 3.16, a C++17 compiler, and OpenCV ≥ 4.6 (`core imgproc imgcodecs dnn`;
4.6 is the floor `cpp/CMakeLists.txt` actually declares).
The ONNX Runtime **SDK tarball** is committed at
`cpp/third_party/onnxruntime-linux-x64-1.20.0.tgz`; the extracted tree is **not**
tracked (it's gitignored) — `cmake` auto-extracts it on first configure.

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release   # auto-extracts the SDK if missing
cmake --build cpp/build --config Release -j
```

To pre-extract or debug an extraction, run the standalone helper instead:

```bash
bash cpp/third_party/setup-onnxruntime.sh
```

Both extract the tarball (which unpacks to a versioned top folder
`onnxruntime-linux-x64-1.20.0/`) and rename it to `third_party/onnxruntime/`.
On WSL-on-Windows-drive mounts (`/mnt/...`) and some NTFS mounts, `tar` can't
create the `.so` symlinks the tarball carries — both the CMake auto-extract and
the script tolerate that and materialize `libonnxruntime.so` / `.so.1` as real
copies of the versioned lib so the link step and runtime RPATH load still resolve.

To use a system-installed ORT instead of the vendored one, point `ONNXRUNTIME_DIR`
at a path containing `include/onnxruntime_cxx_api.h` + `lib/libonnxruntime.so`
(Linux) or `lib/onnxruntime.lib` + the `onnxruntime.dll` (Windows). On Linux,
`BUILD_RPATH`/`INSTALL_RPATH` point at the resolved SDK's `lib/` —
`third_party/onnxruntime/lib` for the vendored fallback, or
`$ONNXRUNTIME_DIR/lib` when the env var is set — so the `libonnxruntime.so`
resolves next to the binary. On Windows, the `.dll` is loaded
at runtime via `PATH` (put the SDK's `lib/` on `PATH`, or copy the DLL next to the exe).

Dev: `-DYOLOV8_SANITIZE=ON` enables ASan + UBSan on Linux.

## Run

```bash
# Single image, annotated output
./cpp/build/onnxruntime/ort_cpp \
    -m models/yolov8s_fp32.onnx -y data/data.yaml \
    -i data/images/val/000000001532.jpg -o results/out.jpg

# Benchmark — single image (falls back to the default sample if no -i/-d given)
./cpp/build/onnxruntime/ort_cpp -m models/yolov8s_fp32.onnx -y data/data.yaml \
    -i data/images/val/000000001532.jpg --benchmark 50 --imgsz 640 \
    --save-json results/cpp_bench.json
# Benchmark — multi-image (cycles over a dir; --max-images caps the loaded set;
# images are decoded before the timed loop, so the timings cover detect() only)
./cpp/build/onnxruntime/ort_cpp -m models/yolov8s_fp32.onnx -y data/data.yaml \
    -d data/images/val --benchmark 200 --max-images 50 --save-json results/cpp_bench.json

# Directory batch, parallel workers
./cpp/build/onnxruntime/ort_cpp -m models/yolov8s_fp32.onnx -y data/data.yaml \
    -d data/images/val --workers 4 -o results/batch

# --max-images caps the --dir set in EVERY --dir mode (benchmark and batch);
# 0 = unlimited (the default). The sorted filename order is kept, so the
# capped prefix is deterministic.
```

The default `--model` is `models/yolov8s_fp32.onnx` (the Python `export` artifact —
`models/yolov8s.onnx` does not exist). `--yaml` resolves class names; on any failure the
driver falls back to a hardcoded 12-class COCO-subset list so the demo runs out of the box.

## Thread safety (directory batch)

`--workers N` runs a worker pool of N threads. **Each worker thread owns its own `YOLOv8`
instance** (one ORT session + one preprocessor buffer per thread). The pool passes each task
its owning thread's `worker_id` (the thread's index 0..N-1); a worker
runs one task at a time, so each detector is touched by exactly one thread. This matters because
a `YOLOv8` instance is **not
reentrant**: `Preprocessor` reuses a single NCHW buffer / letterbox canvas / `Ort::Value` per
`detect()` and `detect()` mutates `profile_result_`. (ORT `Session::Run` itself is thread-safe;
the constraint is the per-instance preprocessor buffer + `profile_result_.`)

## Harness integration

`ort_cpp` is reachable from both Python harnesses through the `ort_cpp:<onnx>` backend
prefix (a thin subprocess wrapper — the exe and Python must run under the **same**
environment, both in WSL/Linux or both native Windows; the exe is a native binary, not
cross-ABI). `resolve_ort_cpp_exe` (`src/benchmark.py`) locates the binary via the
`ORT_CPP_PATH` env var, falling back to `cpp/build/onnxruntime/ort_cpp`; it raises a
"build it first" error when neither exists.

### Benchmark (`python main.py benchmark`)

`--model ort_cpp:models/yolov8s_fp32.onnx` stages the speed-test images into a temp dir
with zero-padded names so the C++ `--dir` loader times the *same* images the Python
backends timed. The timing scopes differ: the C++ benchmark times `detect()` =
preprocess + infer + postprocess + NMS on already-decoded images, while the Python
timed loop includes image decode inside `preprocess` — so the Python rows carry decode
cost the C++ rows do not; the C++ protocol is also per-iteration while the Python loop is
per-sweep, so treat cross-protocol rows as indicative. The warmup counts differ too: the
driver warms a fixed 3 iterations before its timed loop, so the CLI `--warmup` count
applies to the Python rows only. The exe's per-image numbers are mapped into
the summary CSV (sweep-scaled percentiles; peak RSS reported as an absolute value — the C++
process has no Python baseline, `rss_increase_mb` is 0). Batch is **1** (the exe has no
batched-forward path); the requested `--batch-size` is kept as the `requested_batch_size`
metrics key (it is not a summary-CSV column). mAP
`--validation` is unsupported for this backend — there is no Python engine to drive the
native evaluator; the precision signal for the C++ path is the consistency harness.

### Consistency (`python main.py consistency`)

`--model1 models/yolov8s_fp32.onnx --model2 ort_cpp:models/yolov8s_fp32.onnx` runs the
**full-pipeline** comparison: the C++ side re-letterboxes from the image paths with its
own `cpp/common/letterboxInto` (mirroring `src/preprocess.py::letterbox`), so the run
validates the C++ preprocess+forward mirror end-to-end. The wrapper stages each batch into a
temp in-dir and rejects suffixes outside `.jpg/.jpeg/.png/.bmp` at staging time (the same
contract as the benchmark's staging — the C++ `--dir` loader would silently drop them). It shells
out to `--dump-raw-dir` once per batch, loads the `.npy`
files back in sorted order, and `np.concatenate`s them to `(bs, C, N)` — the same shape
`ort_forward` produces, so `compare_tensors` and the detection-mode gates run unchanged.
Thread knobs (`--intra-op-threads` / `--inter-op-threads`) are passed from
`_thread_knobs()`; note the Python comparison session is built without
`SessionOptions` (ORT defaults), so the two sides do not share one
operator-scheduling/reduction order.

### `--dump-raw-dir` (the C++ mode the consistency wrapper drives)

```bash
ort_cpp --dump-raw-dir <out_dir> --dir <in_dir> \
    -m models/yolov8s_fp32.onnx --imgsz 640 --intra-op-threads 4 --inter-op-threads 2
```

Loads the model **once**, iterates the (sorted) images in `<in_dir>`, runs
`YOLOv8::forwardRaw` (letterbox + `Session::Run`, no decode/NMS), and writes each raw
output to `<out_dir>/<stem>.npy` — the input's stem with its extension stripped
(little-endian float32, shape `(1, 4+nc, 8400)`).
The npy writer is hand-rolled (no numpy dep on the C++ side). Requires `--dir`;
ignores `--conf`/`--iou` (raw output is pre-NMS).

## Recorded runs

All figures below come from full recorded runs on the validated host
(2026-10-26, WSL2, 13th Gen Intel Core i5-13420H, 12 logical cores):
Python 3.12.3, ONNX Runtime 1.26.0, torch 2.6.0+cpu, imgsz 640, batch 1,
images drawn from the COCO-subset val split (397 available). Thread knobs
were intra=4 / inter=2 — a derived configuration (`_thread_knobs()` =
`min(4, ncpu)` intra / `2` inter on this host), not a probed measurement —
passed to the C++ exe in both runs. Scope note: in the consistency run this
covers the `ort_cpp` side only; the Python comparison session is built
without `SessionOptions` and uses ORT defaults (see
[Harness integration](#harness-integration)).

### Consistency (`models/yolov8s_fp32.onnx` vs `ort_cpp:models/yolov8s_fp32.onnx`)

Tensor mode, strict pair (`--atol 1e-4 --rtol 1e-3`), 16 stratified-sampled
val images, bs=1. Recorded command (only `--model1/--model2/--atol/--rtol/
--max-images` are non-default):

```bash
python main.py consistency \
    --model1 models/yolov8s_fp32.onnx \
    --model2 ort_cpp:models/yolov8s_fp32.onnx \
    --atol 1e-4 --rtol 1e-3 --max-images 16
```

| metric | value |
| --- | --- |
| fail_rate | 0.00% (0/16 batches) |
| cosine_similarity_mean | 0.9999999999999907 |
| mean_diff_mean | 3.83e-06 |
| p95_mean / p99_mean | 3.05e-05 / 6.32e-05 |
| max_diff_mean | 1.14e-03 |
| std_diff_mean | 1.85e-05 |

### Benchmark

`--warmup 10 --runs 25 --max-images 16`. The two rows carry different timing
protocols (see [Harness integration](#harness-integration)): the Python row
times preprocess (including image decode) → forward → NMS per sweep; the
`ort_cpp` row times `detect()` (preprocess + infer + postprocess + NMS) per
iteration on pre-decoded images, with per-image numbers scaled to sweep units
for the summary columns. Cross-protocol rows are indicative. The warmup differs per
row too: `--warmup 10` covers the Python row, while the `ort_cpp` driver warms a fixed
3 iterations before its timed loop. Percentiles are
per-sweep (16 images); peak RSS is per protocol — the Python row reports the
in-process delta from a pre-load baseline, the `ort_cpp` row reports the
separate C++ process's absolute peak.

| row | image_s | FPS | p50 (ms, sweep) | p95 | p99 | peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| `onnx_fp32` (Python, CPUExecutionProvider) | 0.193 | 5.17 | 2863.1 | 3241.1 | 3561.7 | 670.8 MB (Δ 213.6 MB) |
| `ort_cpp` | 0.194 | 5.15 | 2761.5 | 2992.4 | 3092.0 | 203.8 MB (absolute) |
