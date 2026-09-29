#include "postprocessor.h"

#include <stdexcept>

std::vector<Detection> Postprocessor::process(
    const Ort::Value& output_tensor,
    const cv::Mat& image,
    const PreprocessResult& pre,
    const std::vector<std::string>& class_names,
    float conf_threshold,
    float iou_threshold
) {
    // Resolve the raw pointer + shape, then delegate every bit of numerics to the shared
    // common core. This keeps the ORT-specific surface to exactly "read the tensor"; the
    // decode/scale-back/clamp/NMS path is identical for any C++ backend that can produce a
    // flat float* + a 3-D shape.
    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error(
            "Expected a float32 YOLOv8 output tensor (element type "
            + std::to_string(tensor_info.GetElementType()) + ")"
        );
    }
    const auto shape = tensor_info.GetShape();

    int num_boxes = 0;
    int num_attrs = 0;
    bool transpose = false;
    parseOutputShape(shape, num_boxes, num_attrs, transpose);

    RawOutputView view;
    view.data      = output_tensor.GetTensorData<float>();
    view.num_boxes = num_boxes;
    view.num_attrs = num_attrs;
    view.transpose = transpose;

    return decode_nms(
        view, pre, image, class_names,
        conf_threshold, iou_threshold,
        /*max_det=*/300
    );
}
