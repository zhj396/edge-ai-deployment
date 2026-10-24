#pragma once

#include <string>
#include <opencv2/opencv.hpp>

// One detection after class-wise NMS. Coordinates are in the original
// image's pixel space (not the letterboxed input space).
struct Detection {
    cv::Rect box;
    float conf;
    int class_id;
    std::string class_name;
};

// Output of Preprocessor — the letterbox transform parameters the
// postprocessor needs to map boxes back to the original image.
struct PreprocessResult {
    float scale = 1.0f;      // letterbox scale factor
    cv::Point2f pad{0.f, 0.f};  // single-side padding (x, y) — the floored half-pad
                                // int(dw/2) (matches the ROI placement and the pad
                                // Python's pipeline passes to scale_boxes)
    int orig_w = 0;          // original image width  (pre-letterbox)
    int orig_h = 0;          // original image height (pre-letterbox)
};

// Per-stage wall-clock breakdown for a single detect() call.
struct ProfileResult {
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
};
