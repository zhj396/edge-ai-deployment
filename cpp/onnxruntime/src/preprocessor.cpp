#include "preprocessor.h"

#include <cstring>

Preprocessor::Preprocessor(int input_size, const Ort::MemoryInfo& memory_info)
    : input_size_(input_size)
{
    input_tensor_size_ = static_cast<size_t>(3) * input_size_ * input_size_;
    buffer_.resize(input_tensor_size_);

    std::vector<int64_t> shape = {1, 3, input_size_, input_size_};
    // CreateTensor takes the MemoryInfo by const ref and copies what it needs into the
    // OrtValue — we do not retain memory_info beyond this ctor (it's move-only).
    tensor_ = Ort::Value::CreateTensor<float>(
        memory_info,
        buffer_.data(),
        input_tensor_size_,
        shape.data(),
        shape.size()
    );

    // Pre-allocate the BGR letterbox canvas. Written into in-place each frame by
    // common::letterboxInto (no allocation per call).
    bgr_buffer_.resize(static_cast<size_t>(input_size_) * input_size_ * 3);
    letterbox_bgr_ = cv::Mat(input_size_, input_size_, CV_8UC3, bgr_buffer_.data());
}

Ort::Value& Preprocessor::process(const cv::Mat& image, PreprocessResult& out) {
    // Step 1: Letterbox into the pre-allocated BGR canvas via the shared common core.
    //         LetterboxParams (scale/pad) is written by common::letterboxInto and forwarded
    //         into the PreprocessResult the postprocessor reads — so the inverse transform
    //         in common::decode_nms stays in lockstep with the forward transform applied here.
    LetterboxParams lb;
    letterboxInto(image, letterbox_bgr_, input_size_, lb);
    out.scale = lb.scale;
    out.pad.x = lb.pad_x;
    out.pad.y = lb.pad_y;
    out.orig_w = lb.orig_w;
    out.orig_h = lb.orig_h;

    // Step 2: HWC BGR uchar -> CHW RGB float32 / 255 (SIMD via OpenCV).
    //         swapRB=true flips BGR->RGB. mean=0 (YOLO expects raw 0..1 range,
    //         not ImageNet mean-subtracted).
    cv::Mat blob = cv::dnn::blobFromImage(
        letterbox_bgr_,
        1.0 / 255.0,
        cv::Size(input_size_, input_size_),  // already letterboxed to this
        cv::Scalar(),  // mean
        true,          // swapRB
        false,         // crop
        CV_32F
    );

    // Step 3: copy into our pre-allocated buffer (wraps Ort::Value). blobFromImage
    //         allocates its own Mat; we can't keep it alive across calls, so we copy.
    //         ~4.7MB memcpy at 640x640.
    std::memcpy(
        buffer_.data(),
        blob.ptr<float>(),
        input_tensor_size_ * sizeof(float)
    );

    return tensor_;
}
