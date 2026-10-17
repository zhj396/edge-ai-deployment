#include "letterbox.h"

#include <algorithm>

// Backend-agnostic letterbox implementation shared by every C++ backend.
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

    // Round-to-nearest the resized dims.
    const int resized_w = cv::saturate_cast<int>(
        static_cast<float>(img_w) * out.scale);
    const int resized_h = cv::saturate_cast<int>(
        static_cast<float>(img_h) * out.scale);
    // Float half-pad: consumers (ROI placement, PreprocessResult) floor it to
    // int(dw/2).
    out.pad_x = static_cast<float>(input_size - resized_w) / 2.0f;
    out.pad_y = static_cast<float>(input_size - resized_h) / 2.0f;

    // Fill the whole letterbox canvas with YOLO default gray (114,114,114).
    dst.setTo(cv::Scalar(114, 114, 114));

    if (resized_w <= 0 || resized_h <= 0) {
        return;
    }

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

    // Copy the resized image into the centered ROI: place at floor(pad/2),
    // remainder on right/bottom.
    cv::Mat roi = dst(cv::Rect(cvFloor(out.pad_x), cvFloor(out.pad_y), resized_w, resized_h));
    resized.copyTo(roi);
}
