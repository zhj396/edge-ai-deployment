#include "nms_decode.h"

#include <algorithm>
#include <stdexcept>

// Backend-agnostic YOLOv8 head decode + NMS shared by every C++ backend; the
// decode / scale-back / clamp / NMS numerics match ``src/postprocess.py``.
// See the header for the parity notes and the NMS-divergence
// caveat (cv::dnn::NMSBoxes approximates ultralytics non_max_suppression; the consistency
// harness compares raw tensors, not post-NMS boxes, so this is visualization-only).

void parseOutputShape(
    const std::vector<int64_t>& shape,
    int& num_boxes,
    int& num_attributes,
    bool& transpose
) {
    if (shape.size() != 3) {
        throw std::runtime_error(
            "Unsupported YOLOv8 output rank: expected 3, got "
            + std::to_string(shape.size())
        );
    }

    const int dim1 = static_cast<int>(shape[1]);
    const int dim2 = static_cast<int>(shape[2]);

    // [1, 4+nc, 8400]  -> attribute-major, channels < boxes  (transpose)
    // [1, 8400, 4+nc]  -> box-major,      boxes < channels
    if (dim1 < dim2) {
        transpose = true;
        num_attributes = dim1;
        num_boxes      = dim2;
    } else {
        transpose = false;
        num_boxes      = dim1;
        num_attributes = dim2;
    }
}

std::vector<Detection> decode_nms(
    const RawOutputView& out,
    const PreprocessResult& pre,
    const cv::Mat& image,
    const std::vector<std::string>& class_names,
    float conf_threshold,
    float iou_threshold,
    int max_det
) {
    std::vector<Detection> detections;

    const float* output    = out.data;
    const int num_boxes    = out.num_boxes;
    const int num_attrs    = out.num_attrs;
    const bool transpose   = out.transpose;
    const int num_classes  = num_attrs - 4;
    const float img_w      = static_cast<float>(image.cols);
    const float img_h      = static_cast<float>(image.rows);

    if (num_classes <= 0) {
        return detections;
    }

    // Per-class candidate buffers; class-wise NMS is applied after the conf filter.
    std::vector<std::vector<cv::Rect>> class_boxes(num_classes);
    std::vector<std::vector<float>>    class_scores(num_classes);

    for (int i = 0; i < num_boxes; ++i) {
        float cx, cy, w, h;
        if (transpose) {
            // Layout: [1, 4+nc, 8400] — attribute-major, box at indices 0..3
            cx = output[0 * num_boxes + i];
            cy = output[1 * num_boxes + i];
            w  = output[2 * num_boxes + i];
            h  = output[3 * num_boxes + i];
        } else {
            // Layout: [1, 8400, 4+nc] — box-major
            const int offset = i * num_attrs;
            cx = output[offset + 0];
            cy = output[offset + 1];
            w  = output[offset + 2];
            h  = output[offset + 3];
        }

        // Argmax over class scores (objectness is folded into the cls scores in
        // YOLOv8's exported head — no separate obj channel).
        float max_score = 0.0f;
        int   class_id  = -1;
        for (int c = 0; c < num_classes; ++c) {
            const float score = transpose
                ? output[(4 + c) * num_boxes + i]
                : output[i * num_attrs + 4 + c];
            if (score > max_score) {
                max_score = score;
                class_id  = c;
            }
        }
        if (max_score < conf_threshold) continue;
        if (class_id < 0 || class_id >= num_classes) continue;

        // Inverse letterbox: undo (scale, pad), then clamp to original image bounds.
        // Matches scale_boxes(ratio_pad=((r,r),(pad_x,pad_y))) + post_process's
        // explicit [0, w-1]/[0, h-1] clamp.
        float x1 = (cx - w * 0.5f - static_cast<float>(pre.pad.x)) / pre.scale;
        float y1 = (cy - h * 0.5f - static_cast<float>(pre.pad.y)) / pre.scale;
        float x2 = (cx + w * 0.5f - static_cast<float>(pre.pad.x)) / pre.scale;
        float y2 = (cy + h * 0.5f - static_cast<float>(pre.pad.y)) / pre.scale;

        x1 = std::clamp(x1, 0.0f, img_w - 1.0f);
        y1 = std::clamp(y1, 0.0f, img_h - 1.0f);
        x2 = std::clamp(x2, 0.0f, img_w - 1.0f);
        y2 = std::clamp(y2, 0.0f, img_h - 1.0f);

        cv::Rect box(
            static_cast<int>(x1),
            static_cast<int>(y1),
            std::max(1, static_cast<int>(x2 - x1)),
            std::max(1, static_cast<int>(y2 - y1))
        );

        class_boxes[class_id].push_back(box);
        class_scores[class_id].push_back(max_score);
    }

    // Class-wise NMS (the 5-arg cv::dnn::NMSBoxes overload exists since
    // OpenCV 3.x), then a max_det cap for behavioral parity
    // with ultralytics non_max_suppression (which keeps the top max_det after a final sort).
    // NOTE: cv::dnn::NMSBoxes is an approximation of ultralytics NMS — see header.
    struct ScoredDetection {
        Detection det;
        float score;
    };
    std::vector<ScoredDetection> kept;
    kept.reserve(static_cast<size_t>(max_det));

    for (int c = 0; c < num_classes; ++c) {
        if (class_boxes[c].empty()) continue;

        std::vector<int> indices;
        cv::dnn::NMSBoxes(
            class_boxes[c],
            class_scores[c],
            conf_threshold,
            iou_threshold,
            indices
        );

        for (int idx : indices) {
            Detection det;
            det.box      = class_boxes[c][idx];
            det.conf     = class_scores[c][idx];
            det.class_id = c;
            if (c < static_cast<int>(class_names.size())) {
                det.class_name = class_names[c];
            } else {
                det.class_name = "class_" + std::to_string(c);
            }
            kept.push_back({det, class_scores[c][idx]});
        }
    }

    // Cap at max_det, highest score first (mirrors ultralytics' final max_det pass).
    if (static_cast<int>(kept.size()) > max_det) {
        std::partial_sort(
            kept.begin(), kept.begin() + max_det, kept.end(),
            [](const ScoredDetection& a, const ScoredDetection& b) {
                return a.score > b.score;
            });
        kept.resize(static_cast<size_t>(max_det));
    }

    detections.reserve(kept.size());
    for (const auto& s : kept) {
        detections.push_back(s.det);
    }
    return detections;
}
