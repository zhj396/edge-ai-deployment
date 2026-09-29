#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "types.h"
#include "nms_decode.h"

// ORT Postprocessor: a thin shim that pulls a ``float*`` + shape out of an ``Ort::Value``
// and hands them to the backend-agnostic ``common::decode_nms``. The decode / scale-back /
// clamp / NMS numerics live in ``common/nms_decode`` (single-sourced across C++ backends,
// mirroring ``src/postprocess.py``); this class only owns the ORT-specific tensor read.
//
// YOLOv8 exports its output in EITHER [1, 4+nc, 8400] (post-2023.10) or [1, 8400, 4+nc]
// (older) layout — ``common::parseOutputShape`` detects which; this class is layout-agnostic.
class Postprocessor {
public:
    Postprocessor() = default;

    // Decode output_tensor into detections in the original image's pixel space (the
    // preprocessor's letterbox transform is reversed inside ``decode_nms``). Conf threshold
    // is applied pre-NMS; IoU threshold drives class-wise ``cv::dnn::NMSBoxes``; a
    // ``max_det`` cap (300) is applied for parity with ultralytics non_max_suppression.
    std::vector<Detection> process(
        const Ort::Value& output_tensor,
        const cv::Mat& image,
        const PreprocessResult& pre,
        const std::vector<std::string>& class_names,
        float conf_threshold,
        float iou_threshold
    );
};
