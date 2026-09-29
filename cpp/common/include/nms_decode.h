#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "types.h"

// Backend-agnostic YOLOv8 head decode + class-wise NMS — the C++ mirror of
// ``src/postprocess.py``. Operates on a raw ``const float*`` + a parsed shape, so it has no
// ORT/TensorRT types and is reused by every C++ backend: the backend's Postprocessor just
// pulls a ``float*``/shape out of its own tensor type and hands it to ``decode_nms``.
//
// NUMERIC PARITY (vs the Python path): box decode (xywh -> xyxy), inverse-letterbox
// scale-back ``(x - pad) / scale``, and the post-scale clamp to ``[0, w-1] / [0, h-1]`` match
// ``ultralytics.utils.ops.scale_boxes`` + the explicit ``[0, w-1]`` clamp in
// ``src/postprocess.py::post_process`` exactly. The NMS itself uses
// ``cv::dnn::NMSBoxes`` as an *approximation* of ``ultralytics.utils.nms.non_max_suppression``;
// the two are NOT bit-identical (different IoU tie-breaking, and ultralytics applies a final
// ``max_det`` cap + re-sort). The divergence is acceptable here because the cross-backend
// **consistency harness compares raw forward tensors, not post-NMS detections** — the C++ app's
// drawn boxes are a standalone visualization, not a consistency input. A ``max_det`` cap is
// applied for behavioral parity (default 300, matching ultralytics).

// Non-owning view over a decoded YOLOv8 output tensor.
struct RawOutputView {
    const float* data = nullptr;  // raw flat data (caller owns / must outlive the view)
    int num_boxes    = 0;          // anchor count (e.g. 8400 @ 640px)
    int num_attrs    = 0;          // 4 + nc
    bool transpose  = false;       // true => [1, 4+nc, num_boxes]; false => [1, num_boxes, 4+nc]
};

// Detect the YOLOv8 output layout from its 3-D shape and populate a RawOutputView's
// (num_boxes, num_attrs, transpose). ``shape[1] < shape[2]`` => attribute-major (transpose).
// This dimension comparison resolves the standard export unambiguously
// (16 vs 8400 @ 640px); it is coarser than ``src/postprocess.py::
// _ensure_4nc_first``'s magnitude rule, which also guards small N tensors.
void parseOutputShape(
    const std::vector<int64_t>& shape,
    int& num_boxes,
    int& num_attributes,
    bool& transpose
);

// Decode boxes + argmax class scores, conf-filter, inverse-letterbox scale-back, clamp,
// class-wise ``cv::dnn::NMSBoxes``, and a ``max_det`` cap. Returns detections in the
// ORIGINAL image's pixel space (clamped to ``[0, w-1]``/``[0, h-1]``).
std::vector<Detection> decode_nms(
    const RawOutputView& out,
    const PreprocessResult& pre,
    const cv::Mat& image,
    const std::vector<std::string>& class_names,
    float conf_threshold,
    float iou_threshold,
    int max_det = 300
);
