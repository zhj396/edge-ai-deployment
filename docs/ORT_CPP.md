# ONNX Runtime C++ backend (`cpp/`)

The 5th inference path (after PT / ONNX-FP32 / ONNX-INT8 / OpenVINO): a standalone C++
YOLOv8s inference driver built on the ONNX Runtime C++ API. C++ can't `import` the Python
pipeline, so `cpp/` is an additive tree beside the layered Python package — it does **not**
replace or reorganize `cli/`+`src/`+`utils/`.

> Unlike the Python backends, the C++ backend is **not** wired into `src/consistency.py`
> yet. The plan for that is a thin Python wrapper (subprocess or `pybind11`)
> exposing the C++ forward as a callable — the comparison logic itself is not forked.

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
`cpp/onnxruntime/`. A future `cpp/tensorrt/` backend reuses `cpp/common/` unchanged: it links
`ort_core_common` and hands `decode_nms` a raw `float*` + shape pulled from a TRT buffer.
This mirrors the Python project's hook — a *shared preprocess/postprocess core,
consistency-tested across backends* — and keeps that core single-sourced in C++ too.

**SDK path.** The ONNX Runtime SDK lives at `cpp/third_party/onnxruntime/` — *not*
`cpp/onnxruntime/`. The latter path is reserved for the backend *source*. `ONNXRUNTIME_DIR`
env var overrides the SDK location. Note the C++ tree vendors ORT **1.20.0** while the
Python path pins `onnxruntime==1.26.0` (`requirements-cpu.txt`) — the runtimes are
independent installs.

## Numerical consistency vs the Python path

The C++ core is written to be numerically consistent with the Python preprocess/postprocess
(within tolerance — two implementation notes below):

- **Letterbox.** `common/letterbox.cpp::letterboxInto` mirrors `src/preprocess.py::letterbox`.
  Resized dimensions are **rounded** (`cv::saturate_cast<int>` ≡ `cvRound`, matching Python's
  `int(round(shape * r))`) so both paths letterbox to identical sizes for non-integer scales.
  The gray fill (114,114,114), `INTER_LINEAR` resize, and centered-ROI copy all match.
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
  A `max_det=300` cap is applied for behavioral parity. This is acceptable here because the
  cross-backend **consistency harness compares raw forward tensors, not post-NMS detections**;
  the C++ NMS only affects the standalone app's drawn boxes, not a consistency input.
  Reimplementing ultralytics NMS in C++ is explicitly out of scope.

## Build

Requires CMake ≥ 3.16, a C++17 compiler, and OpenCV ≥ 4.6 (`core imgproc imgcodecs dnn`).
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
`BUILD_RPATH`/`INSTALL_RPATH` point at `third_party/onnxruntime/lib` so the
`libonnxruntime.so` resolves next to the binary. On Windows, the `.dll` is loaded
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
# Benchmark — multi-image (cycles over a dir, capped by --max-images; per-image
# decode and cold-cache input traffic are included in the timings)
./cpp/build/onnxruntime/ort_cpp -m models/yolov8s_fp32.onnx -y data/data.yaml \
    -d data/images/val --benchmark 200 --max-images 50 --save-json results/cpp_bench.json

# Directory batch, parallel workers
./cpp/build/onnxruntime/ort_cpp -m models/yolov8s_fp32.onnx -y data/data.yaml \
    -d data/images/val --workers 4 -o results/batch
```

The default `--model` is `models/yolov8s_fp32.onnx` (the Python `export` artifact —
`models/yolov8s.onnx` does not exist). `--yaml` resolves class names; on any failure the
driver falls back to a hardcoded 12-class COCO-subset list so the demo runs out of the box.

## Thread safety (directory batch)

`--workers N` runs a worker pool of N threads. **Each worker thread owns its own `YOLOv8`
instance** (one ORT session + one preprocessor buffer per thread). The pool passes each task
its owning thread's `worker_id` (the thread's index 0..N-1, *not* `i % n_workers`); a worker
runs one task at a time, so each detector is touched by exactly one thread. This matters because
a `YOLOv8` instance is **not
reentrant**: `Preprocessor` reuses a single NCHW buffer / letterbox canvas / `Ort::Value` per
`detect()` and `detect()` mutates `profile_result_` — two tasks tagged for the same worker
would run concurrently against `worker_detectors[k]`, because the shared queue lets any free
thread grab any task. (ORT `Session::Run` itself is thread-safe; the constraint is the
per-instance preprocessor buffer + `profile_result_`.)

## Consistency-harness integration (future)

`src/consistency.py` compares backends by **forward fn**. Wiring the C++ backend means a
small Python adapter exposing a forward callable over the C++ preprocess→`Session::Run` path
(subprocess that reads a raw input tensor and dumps the raw output tensor, or a `pybind11`
module). The adapter must reuse `common/letterboxInto` for the input (so it matches the
Python preprocess) and return the **raw** `Ort::Value` (pre-NMS) — `compare_tensors` /
the conf-cliff gate do the rest. Do **not** fork the comparison logic or thread NMS into it.
