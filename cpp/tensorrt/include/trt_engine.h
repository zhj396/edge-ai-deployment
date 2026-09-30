// trt_engine.h — in-process TensorRT 10.x forward-only accelerator (pybind11).
//
// Thin port of the standalone reference at yolov8s_trt_cpp/src/yolo_engine.cpp
// (TRT-10 tensor-name API + CUDA Runtime H2D/exe/D2H), stripped to a forward-
// only accelerator: no preprocess, no postprocess, no NMS, no per-thread
// context pool (the Python caller is single-threaded). It deserialises a
// .engine, runs enqueueV3 (the C++ API; the Python binding names it
// execute_async_v3), and hands the raw [bs, 4+nc, 8400] head back to Python. Preprocess/postprocess stay Python (reuse src/preprocess
// + src/postprocess), so the C++ module never links ultralytics.
//
// CUDA Runtime API only (cudaMalloc / cudaMemcpyAsync / cudaStreamSynchronize
// / cudaFree) — the Runtime API auto-manages the primary context, so there is
// no cuCtxPushCurrent/cuCtxPopCurrent (the Python TensorRTEngine's push/pop
// is a Driver-API artifact via cuda-python lean bindings; C++ has no such
// constraint). The engine is LOAD-ONLY: build stays in Python
// (build_tensorrt_engine), so this links nvinfer only, no nvonnxparser.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_cpp {

class TrtEngine {
public:
    /// Deserialise `engine_path` on `device_id`, allocate a single execution
    /// context + device/host buffers sized for `max_batch` (dynamic dims in
    /// the engine's profile are resolved to max_batch / 4+nc / 8400 so any
    /// 1..max_batch forward fits without re-alloc). Throws std::runtime_error
    /// on any IO / TRT / CUDA failure (pybind11 translates it to RuntimeError).
    TrtEngine(const std::string& engine_path, int device_id = 0,
              int imgsz = 640, int max_batch = 8);
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    /// Run a forward pass. `host_in` is a C-contiguous float32 buffer shaped
    /// `in_shape` (typically {bs, 3, H, W}); the output shape (engine-reported,
    /// runtime-resolved after setInputShape) is written to `out_shape`. Returns
    /// the raw output floats (the caller memcpy's into a numpy array).
    std::vector<float> forward(const float* host_in,
                               const std::vector<int64_t>& in_shape,
                               std::vector<int64_t>& out_shape);

    /// Same as forward, but the timer brackets only cudaStreamSynchronize — the
    /// kernel window (H2D + execute + D2H), free of Python NMS/letterbox. Writes
    /// the elapsed milliseconds to `kernel_ms`. Mirrors Python
    /// TensorRTEngine.kernel_timed_forward exactly.
    std::vector<float> kernel_timed_forward(const float* host_in,
                                            const std::vector<int64_t>& in_shape,
                                            std::vector<int64_t>& out_shape,
                                            double& kernel_ms);

    /// Idempotent teardown: stream -> device buffers. Safe to call from
    /// __exit__ / finally. After release(), forward() throws.
    void release();
    bool released() const noexcept;

private:
    void load_engine(const std::string& path);
    void discover_io();
    void allocate_buffers();
    void set_tensor_addresses();
    /// Shared H2D -> enqueueV3 -> D2H -> sync body for forward /
    /// kernel_timed_forward. A private MEMBER (not a free function) so it can
    /// access the private Impl; ``kernel_ms`` non-null brackets only the sync.
    std::vector<float> run_forward(const float* host_in,
                                   const std::vector<int64_t>& in_shape,
                                   std::vector<int64_t>& out_shape,
                                   double* kernel_ms);

    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace trt_cpp
