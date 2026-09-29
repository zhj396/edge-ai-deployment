#include "letterbox.h"

#include <algorithm>

// Backend-agnostic letterbox implementation shared by every C++ backend; the
// numerics match ``src/preprocess.py::letterbox``.
void letterboxInto(
    const cv::Mat& src,
    cv::Mat& dst,
    int input_size,
    LetterboxParams& out
) {
    const int img_w = src.cols;
    const int img_h = src.rows;
    out.orig_w = img_w;
    out.orig_h = img_h;

    out.scale = std::min(
        static_cast<float>(input_size) / static_cast<float>(img_w),
        static_cast<float>(input_size) / static_cast<float>(img_h)
    );

    // Round, matching Python's int(round(shape * r)) — cv::saturate_cast<int>
    // == cvRound (round-half-to-even) — so the resized dims match the Python
    // letterbox for non-integer scales.
    const int resized_w = cv::saturate_cast<int>(
        static_cast<float>(img_w) * out.scale);
    const int resized_h = cv::saturate_cast<int>(
        static_cast<float>(img_h) * out.scale);
    out.pad_x = (input_size - resized_w) / 2;
    out.pad_y = (input_size - resized_h) / 2;

    // Fill the whole letterbox canvas with YOLO default gray (114,114,114).
    dst.setTo(cv::Scalar(114, 114, 114));

    if (resized_w <= 0 || resized_h <= 0) {
        return;
    }

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

    // Copy the resized image into the centered ROI. The remainder of the gray fill on
    // the right/bottom edges stays (single-side pad == pad_x/pad_y; the other side gets
    // the remainder of (input_size - resized), matching Python's copyMakeBorder split).
    cv::Mat roi = dst(cv::Rect(out.pad_x, out.pad_y, resized_w, resized_h));
    resized.copyTo(roi);
}
