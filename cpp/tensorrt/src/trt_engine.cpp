// trt_engine.cpp — TensorRT 10.x forward-only accelerator (CUDA Runtime API).
//
// TRT-10 tensor-name API + enqueueV3 + cudaMemcpyAsync, thinned to
// forward-only: no preprocess, no postprocess, no context pool.
// Single-threaded Python caller via pybind11.

#include "trt_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

namespace trt_cpp {

namespace {

// Silent TRT logger — TRT requires one at deserialize time; every severity
// is dropped so the pybind11 module never writes to the user's stdout.
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity /*severity*/, const char* /*msg*/) noexcept override {}
};

// Byte size of one element of a TRT data type. Only kFLOAT IO is accepted
// (enforced in discover_io): the forward path memcpys raw float32 bytes in
// both directions, so other IO dtypes would produce garbage.
std::size_t bytes_per_element(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        default:                         return 4;  // unreachable — discover_io() rejects non-float IO
    }
}

// Upper-bound element count of a tensor, resolving dynamic dims (-1) to the
// constructor's max_batch (dim 0) or the YOLOv8 maxima (84 channels on dim 1,
// 8400 anchors on dim 2). Used to size the device/host buffers ONCE at alloc
// so any 1..max_batch forward fits without re-alloc — mirrors Python
// _allocate_buffers' dynamic-dim → maximum resolution.
std::size_t max_elems(const nvinfer1::Dims& dims, int max_batch) {
    std::size_t n = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        int d = dims.d[i];
        if (d < 0) {
            if (i == 0)      d = max_batch;     // batch
            else if (i == 1) d = 84;           // 4 + COCO_MAX_CLASSES
            else if (i == 2) d = 8400;         // anchors
            else             d = 1;           // conservative
        }
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

}  // namespace

struct TrtEngine::Impl {
    TrtLogger trt_log;

    std::unique_ptr<nvinfer1::IRuntime>        runtime;
    std::unique_ptr<nvinfer1::ICudaEngine>     engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    std::string input_name;
    std::string output_name;
    nvinfer1::DataType in_dt  = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType out_dt = nvinfer1::DataType::kFLOAT;

    int channels = 3;
    int height   = 640;
    int width    = 640;
    int max_batch = 8;
    int device_id = 0;

    // Device buffers (Runtime API). d_in/d_out are the addresses bound via
    // setTensorAddress; the same addresses serve every forward (TRT reads the
    // input / writes the output at these pointers on the engine's stream).
    void* d_in  = nullptr;
    void* d_out = nullptr;
    std::size_t d_in_bytes  = 0;
    std::size_t d_out_bytes = 0;

    // Host staging buffer. The H2D copy reads directly from the caller's
    // pybind11 array buffer; h_out is the D2H landing zone, copied into a
    // fresh numpy array per forward. Reused across calls — the returned
    // numpy owns its own buffer (memcpy), no capsule/shared ownership across
    // the boundary.
    std::vector<float> h_out;

    cudaStream_t stream = nullptr;
    bool released = false;
};

TrtEngine::TrtEngine(const std::string& engine_path, int device_id,
                     int imgsz, int max_batch)
    : pimpl_(std::make_unique<Impl>()) {
    pimpl_->device_id = device_id;
    pimpl_->max_batch = std::max(1, max_batch);
    pimpl_->height = imgsz;
    pimpl_->width  = imgsz;

    // Runtime API auto-manages the primary context — cudaSetDevice is the
    // only device-level call needed.
    cudaError_t ce = cudaSetDevice(pimpl_->device_id);
    if (ce != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaSetDevice(") + std::to_string(pimpl_->device_id)
            + ") failed: " + cudaGetErrorString(ce));
    }
    ce = cudaStreamCreateWithFlags(&pimpl_->stream, cudaStreamNonBlocking);
    if (ce != cudaSuccess) {
        throw std::runtime_error(std::string("cudaStreamCreate failed: ")
                                 + cudaGetErrorString(ce));
    }

    // Everything after stream creation can throw; the constructor's destructor
    // does not run on a throw, so the stream is rolled back here.
    try {
        pimpl_->runtime.reset(nvinfer1::createInferRuntime(pimpl_->trt_log));
        if (!pimpl_->runtime) {
            throw std::runtime_error("createInferRuntime failed");
        }

        load_engine(engine_path);
        discover_io();
        allocate_buffers();
        set_tensor_addresses();
    } catch (...) {
        cudaStreamDestroy(pimpl_->stream);
        pimpl_->stream = nullptr;
        throw;
    }
}

void TrtEngine::load_engine(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("Cannot open engine file: " + path);
    }
    const std::size_t sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<char> data(sz);
    if (!f.read(data.data(), static_cast<std::streamsize>(sz))) {
        throw std::runtime_error("Read failed: " + path);
    }
    pimpl_->engine.reset(
        pimpl_->runtime->deserializeCudaEngine(data.data(), sz));
    if (!pimpl_->engine) {
        throw std::runtime_error("deserializeCudaEngine failed: " + path);
    }
    pimpl_->context.reset(pimpl_->engine->createExecutionContext());
    if (!pimpl_->context) {
        throw std::runtime_error("createExecutionContext failed");
    }
}

void TrtEngine::discover_io() {
    // TRT-10 tensor-name API (the legacy binding-index set_binding_address /
    // enqueueV2 was removed upstream — there is no fallback). Discover the
    // single input + single output by IOMode; resolve the static input H/W
    // (dynamic dims are sized at max_batch in allocate_buffers).
    const int nb = pimpl_->engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* name = pimpl_->engine->getIOTensorName(i);
        const auto mode  = pimpl_->engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            pimpl_->input_name = name;
            pimpl_->in_dt = pimpl_->engine->getTensorDataType(name);
            auto dims = pimpl_->engine->getTensorShape(name);
            if (dims.nbDims == 4) {
                pimpl_->channels = dims.d[1] > 0 ? dims.d[1] : 3;
                pimpl_->height   = dims.d[2] > 0 ? dims.d[2] : pimpl_->height;
                pimpl_->width    = dims.d[3] > 0 ? dims.d[3] : pimpl_->width;
            }
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            pimpl_->output_name = name;
            pimpl_->out_dt = pimpl_->engine->getTensorDataType(name);
        }
    }
    if (pimpl_->input_name.empty() || pimpl_->output_name.empty()) {
        throw std::runtime_error("Engine missing input/output tensor");
    }
    // The forward path memcpys raw float32 bytes both ways; an FP16/INT8 IO
    // engine would produce garbage output, so reject it up front. (Compute
    // precision is unaffected — a builder-FP16 engine keeps float32 IO.)
    if (pimpl_->in_dt != nvinfer1::DataType::kFLOAT
            || pimpl_->out_dt != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error(
            "only float32 engine IO is supported; rebuild the engine without "
            "strongly-typed FP16/INT8 IO (in_dt=" + std::to_string(static_cast<int>(pimpl_->in_dt))
            + ", out_dt=" + std::to_string(static_cast<int>(pimpl_->out_dt)) + ")");
    }

    // Resolve the per-forward batch bound from the serialized engine (mirrors
    // Python TensorRTEngine._load_engine): the optimization profile's MAX for
    // a dynamic-batch engine, 1 for a static engine (the batch=1 build route
    // carries no profile). allocate_buffers sizes against this bound and
    // run_forward rejects batches above it; the Python wrapper reads the
    // bound back for _effective_batch. A dynamic engine without a profile
    // keeps the constructor's bound.
    const auto in_dims = pimpl_->engine->getTensorShape(
        pimpl_->input_name.c_str());
    if (in_dims.nbDims >= 1 && in_dims.d[0] > 0) {
        if (in_dims.d[0] != 1) {
            throw std::runtime_error(
                "Static engine input batch is " + std::to_string(in_dims.d[0])
                + "; only a batch=1 static engine is supported.");
        }
        pimpl_->max_batch = 1;
    } else if (pimpl_->engine->getNbOptimizationProfiles() > 0) {
        const auto prof_max = pimpl_->engine->getProfileShape(
            pimpl_->input_name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        if (prof_max.nbDims >= 1 && prof_max.d[0] > 0) {
            pimpl_->max_batch = std::min(pimpl_->max_batch,
                                         static_cast<int>(prof_max.d[0]));
        }
    }
}

void TrtEngine::allocate_buffers() {
    // Size at the max_batch upper bound so any 1..max_batch forward fits
    // without re-alloc (mirrors Python _allocate_buffers). h_out is reused
    // across calls; the returned numpy owns its own buffer.
    const auto in_dims  = pimpl_->engine->getTensorShape(pimpl_->input_name.c_str());
    const auto out_dims = pimpl_->engine->getTensorShape(pimpl_->output_name.c_str());
    const std::size_t in_elems  = max_elems(in_dims,  pimpl_->max_batch);
    const std::size_t out_elems = max_elems(out_dims, pimpl_->max_batch);

    pimpl_->d_in_bytes  = in_elems  * bytes_per_element(pimpl_->in_dt);
    pimpl_->d_out_bytes = out_elems * bytes_per_element(pimpl_->out_dt);
    pimpl_->h_out.assign(out_elems, 0.0f);

    cudaError_t ce = cudaMalloc(&pimpl_->d_in, pimpl_->d_in_bytes);
    if (ce != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc (in) failed: ")
                                 + cudaGetErrorString(ce));
    }
    ce = cudaMalloc(&pimpl_->d_out, pimpl_->d_out_bytes);
    if (ce != cudaSuccess) {
        cudaFree(pimpl_->d_in);
        pimpl_->d_in = nullptr;
        throw std::runtime_error(std::string("cudaMalloc (out) failed: ")
                                 + cudaGetErrorString(ce));
    }
}

void TrtEngine::set_tensor_addresses() {
    if (!pimpl_->context->setTensorAddress(pimpl_->input_name.c_str(),
                                           pimpl_->d_in)) {
        throw std::runtime_error("setTensorAddress failed for input tensor: "
                                 + pimpl_->input_name);
    }
    if (!pimpl_->context->setTensorAddress(pimpl_->output_name.c_str(),
                                           pimpl_->d_out)) {
        throw std::runtime_error("setTensorAddress failed for output tensor: "
                                 + pimpl_->output_name);
    }
}

// Common H2D -> enqueueV3 -> D2H -> sync body. The kernel-timed variant wraps
// only the sync in a timer (the sync is where H2D+execute+D2H all complete).
// Returns the runtime-resolved output shape + the output floats. A private
// member (not a free function) so it can access the private Impl.
std::vector<float> TrtEngine::run_forward(const float* host_in,
                                          const std::vector<int64_t>& in_shape,
                                          std::vector<int64_t>& out_shape,
                                          double* kernel_ms) {
    auto& impl = *pimpl_;
    if (impl.released) {
        throw std::runtime_error("TrtEngine used after release()");
    }
    if (in_shape.size() != 4) {
        throw std::runtime_error("expected 4-D input {bs, C, H, W}");
    }
    const int bs = static_cast<int>(in_shape[0]);
    if (bs < 1 || bs > impl.max_batch) {
        throw std::runtime_error(
            "batch " + std::to_string(bs) + " outside profile 1.."
            + std::to_string(impl.max_batch));
    }
    // The H2D byte count below is derived from the Impl's resolved C/H/W,
    // so the caller's actual buffer shape must agree — a mismatched spatial
    // shape would otherwise over-read the host buffer in the memcpy.
    if (in_shape[1] != impl.channels || in_shape[2] != impl.height
            || in_shape[3] != impl.width) {
        throw std::runtime_error(
            "input shape (" + std::to_string(in_shape[1]) + ","
            + std::to_string(in_shape[2]) + "," + std::to_string(in_shape[3])
            + ") does not match the engine's resolved CxHxW ("
            + std::to_string(impl.channels) + "," + std::to_string(impl.height)
            + "," + std::to_string(impl.width)
            + "); pass the preprocessed imgsz the engine was built with");
    }
    // Resolve the input shape on this execution context (TRT-10 dynamic-batch:
    // setInputShape per forward). Use plain Dims, the core type setInputShape
    // takes; the legacy Dims4 alias is deprecated in TRT 10.x.
    nvinfer1::Dims in_dims{};
    in_dims.nbDims = 4;
    in_dims.d[0] = bs;
    in_dims.d[1] = impl.channels;
    in_dims.d[2] = impl.height;
    in_dims.d[3] = impl.width;
    if (!impl.context->setInputShape(impl.input_name.c_str(), in_dims)) {
        throw std::runtime_error(
            "setInputShape failed for " + impl.input_name
            + " (shape outside the engine's optimization profile?)");
    }

    const std::size_t in_bytes = static_cast<std::size_t>(bs) * impl.channels
                                 * impl.height * impl.width
                                 * bytes_per_element(impl.in_dt);
    if (in_bytes > impl.d_in_bytes) {
        throw std::runtime_error("input exceeds allocated buffer");
    }
    // H2D: the caller hands a C-contiguous float32 numpy (forcecast in the
    // pybind11 binding guarantees this), so a straight memcpy of in_bytes.
    cudaError_t ce = cudaMemcpyAsync(impl.d_in, host_in, in_bytes,
                                     cudaMemcpyHostToDevice, impl.stream);
    if (ce != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync H2D failed: ")
                                 + cudaGetErrorString(ce));
    }
    // enqueueV3(stream) — TRT-10 async execution (the C++ API name; the Python
    // binding calls it execute_async_v3). Reads d_in (bound via setTensorAddress)
    // on the engine's stream, writes d_out. The legacy enqueueV2 was removed.
    if (!impl.context->enqueueV3(impl.stream)) {
        throw std::runtime_error("enqueueV3 returned false");
    }
    // Runtime-resolved output shape (post-setInputShape): the dynamic batch dim
    // is now concrete bs.
    const auto out_dims = impl.context->getTensorShape(impl.output_name.c_str());
    out_shape.assign(out_dims.d, out_dims.d + out_dims.nbDims);
    std::size_t out_elems = 1;
    // A remaining non-positive dim after setInputShape means the shape never
    // resolved — sizing it to 1 here would desync the memcpy count from the
    // numpy shape downstream, so fail loudly instead.
    for (auto d : out_shape) {
        if (d <= 0) {
            throw std::runtime_error("unresolved dim in output shape of "
                                     + impl.output_name);
        }
        out_elems *= static_cast<std::size_t>(d);
    }
    const std::size_t out_bytes = out_elems * bytes_per_element(impl.out_dt);
    if (out_bytes > impl.d_out_bytes) {
        throw std::runtime_error("output exceeds allocated buffer");
    }
    // D2H into the staging vector, then sync. The sync is the kernel window
    // (H2D + execute + D2H all complete here); kernel_timed_forward brackets
    // only this call (the async work is enqueued above; sync waits for it).
    ce = cudaMemcpyAsync(impl.h_out.data(), impl.d_out, out_bytes,
                         cudaMemcpyDeviceToHost, impl.stream);
    if (ce != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync D2H failed: ")
                                 + cudaGetErrorString(ce));
    }
    // Declare the start point only when timing (mirrors Python
    // kernel_timed_forward: t0 = perf_counter() right before the sync).
    // steady_clock: monotonic (high_resolution_clock aliases system_clock on
    // libstdc++ — an NTP step would corrupt the kernel window).
    auto t0 = kernel_ms
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    ce = cudaStreamSynchronize(impl.stream);
    if (ce != cudaSuccess) {
        throw std::runtime_error(std::string("cudaStreamSynchronize failed: ")
                                 + cudaGetErrorString(ce));
    }
    if (kernel_ms) {
        auto t1 = std::chrono::steady_clock::now();
        *kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return std::vector<float>(impl.h_out.begin(),
                              impl.h_out.begin() + out_elems);
}

std::vector<float> TrtEngine::forward(const float* host_in,
                                      const std::vector<int64_t>& in_shape,
                                      std::vector<int64_t>& out_shape) {
    return run_forward(host_in, in_shape, out_shape, nullptr);
}

std::vector<float> TrtEngine::kernel_timed_forward(
        const float* host_in, const std::vector<int64_t>& in_shape,
        std::vector<int64_t>& out_shape, double& kernel_ms) {
    return run_forward(host_in, in_shape, out_shape, &kernel_ms);
}

void TrtEngine::release() {
    if (pimpl_->released) return;
    // Deterministic teardown order: stream -> device buffers. The Runtime API
    // primary context is auto-managed (freed at process exit / device reset),
    // so no explicit context release is needed. Idempotent.
    if (pimpl_->stream) {
        cudaStreamSynchronize(pimpl_->stream);
        cudaStreamDestroy(pimpl_->stream);
        pimpl_->stream = nullptr;
    }
    if (pimpl_->d_in)  { cudaFree(pimpl_->d_in);  pimpl_->d_in  = nullptr; }
    if (pimpl_->d_out) { cudaFree(pimpl_->d_out); pimpl_->d_out = nullptr; }
    pimpl_->context.reset();
    pimpl_->engine.reset();
    pimpl_->runtime.reset();
    pimpl_->released = true;
}

bool TrtEngine::released() const noexcept { return pimpl_->released; }

int TrtEngine::max_batch() const noexcept { return pimpl_->max_batch; }

TrtEngine::~TrtEngine() { release(); }

}  // namespace trt_cpp
