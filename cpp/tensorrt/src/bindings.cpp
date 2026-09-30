// bindings.cpp — pybind11 module _trt_cpp: a thin forward-only TensorRT C++
// accelerator for YOLOv8s. Bound class `TrtEngine` mirrors the forward-only
// subset of Python TensorRTEngine (raw_forward / kernel_timed_forward /
// release). Preprocess/postprocess stay Python; the C++ module only owns the
// GPU forward (deserialize + enqueueV3 + H2D/D2H).
//
// Numpy boundary:
//   - Input  : py::array_t<float, c_style | forcecast> — forces a C-contiguous
//     float32 view. forcecast copies once if the caller hands a non-contiguous
//     or non-float32 array; the deployed path (pre["images"].cpu().numpy())
//     is already C-contiguous float32, so no copy in the hot path.
//   - Output : a fresh py::array_t<float> with the engine-reported out_shape;
//     memcpy from the C++ host staging vector. No capsule / shared ownership:
//     the returned numpy owns its own buffer (the staging vector is reused
//     across calls). Output ~130KB at 8400 anchors, the copy is negligible vs
//     the GPU forward.

#include <cstring>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "trt_engine.h"

namespace py = pybind11;
using trt_cpp::TrtEngine;

static std::vector<int64_t> array_shape(const py::array& a) {
    return {a.shape(), a.shape() + a.ndim()};
}

PYBIND11_MODULE(_trt_cpp, m) {
    m.doc() = "In-process TensorRT C++ forward accelerator for YOLOv8s (pybind11)";

    py::class_<TrtEngine>(m, "TrtEngine")
        .def(py::init<const std::string&, int, int, int>(),
             py::arg("engine_path"), py::arg("device") = 0,
             py::arg("imgsz") = 640, py::arg("max_batch") = 8,
             "Deserialize a .engine and allocate a single execution context + "
             "device/host buffers sized for max_batch. Throws RuntimeError on "
             "IO / TRT / CUDA failure.")
        // raw_forward(batch_np) -> np.ndarray — the consistency + benchmark
        // forward. Consumes the already-preprocessed tensor x (like the
        // Python tensorrt: path, NOT re-preprocess-from-files like ort_cpp).
        .def("raw_forward", [](TrtEngine& self,
              py::array_t<float, py::array::c_style | py::array::forcecast> batch) {
            auto buf = batch.request();
            std::vector<int64_t> out_shape;
            auto out_vec = self.forward(static_cast<float*>(buf.ptr),
                                        array_shape(batch), out_shape);
            py::array_t<float> result(out_shape);
            std::memcpy(result.mutable_data(), out_vec.data(),
                        out_vec.size() * sizeof(float));
            return result;
        }, py::arg("batch"))
        // kernel_timed_forward(batch_np) -> (np.ndarray, float ms) — the timer
        // brackets only cudaStreamSynchronize (the kernel window: H2D + execute
        // + D2H), free of Python NMS/letterbox. Mirrors Python
        // TensorRTEngine.kernel_timed_forward exactly.
        .def("kernel_timed_forward", [](TrtEngine& self,
              py::array_t<float, py::array::c_style | py::array::forcecast> batch) {
            auto buf = batch.request();
            std::vector<int64_t> out_shape;
            double ms = 0.0;
            auto out_vec = self.kernel_timed_forward(
                static_cast<float*>(buf.ptr), array_shape(batch), out_shape, ms);
            py::array_t<float> result(out_shape);
            std::memcpy(result.mutable_data(), out_vec.data(),
                        out_vec.size() * sizeof(float));
            return py::make_tuple(result, ms);
        }, py::arg("batch"))
        .def("release", &TrtEngine::release,
             "Idempotent teardown: stream + device buffers. Safe in finally/__exit__.");
}
