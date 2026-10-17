#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "types.h"
#include "letterbox.h"

// ORT Preprocessor: letterbox (delegated to the backend-agnostic common::letterboxInto)
// + HWC BGR -> CHW RGB float32 / 255 into a pre-allocated ``Ort::Value`` buffer.
//
// The NCHW float buffer is allocated once at construction and reused across every detect()
// call; the intermediate BGR letterbox canvas is also pre-allocated to avoid per-frame
// malloc. The letterbox *math* lives in ``common/letterbox`` (shared across C++ backends,
// numerically consistent with ``src/preprocess.py``); this class owns only the ORT-specific
// tensor wrapping: ``cv::dnn::blobFromImage`` (OpenCV SIMD HWC->CHW + /255 + swapRB) + the
// ``Ort::Value`` view over our pre-allocated buffer. A memcpy bridges blobFromImage's own
// (per-call) Mat into the pre-allocated ORT buffer.
class Preprocessor {
public:
    Preprocessor(int input_size, const Ort::MemoryInfo& memory_info);

    // Letterbox `image` into the internal NCHW float buffer and return a reference to the
    // wrapped Ort::Value. The reference is valid until the next call to process() or until
    // this object is destroyed. Writes the letterbox transform into `out` for the
    // postprocessor's inverse map.
    Ort::Value& process(const cv::Mat& image, PreprocessResult& out);

    size_t tensor_size() const { return input_tensor_size_; }

private:
    int input_size_;
    size_t input_tensor_size_;

    // Pre-allocated NCHW float32 buffer wrapped by tensor_ (Ort::Value).
    std::vector<float> buffer_;
    // memory_info is NOT stored: Ort::MemoryInfo is move-only (copy deleted), and it is
    // only needed at construction to create `tensor_` (CreateTensor copies the info it
    // needs into the OrtValue). `process()` never touches it again.
    Ort::Value tensor_{nullptr};

    // Pre-allocated BGR letterbox canvas (HWC uchar) handed to common::letterboxInto.
    std::vector<uchar> bgr_buffer_;
    cv::Mat letterbox_bgr_;
};
