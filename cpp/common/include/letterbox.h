#pragma once

#include <opencv2/opencv.hpp>

#include "types.h"

// Backend-agnostic letterbox.
//
// Fills ``dst`` (a pre-allocated ``input_size x input_size`` ``CV_8UC3`` buffer owned by the
// caller) with YOLO gray (114,114,114), resizes ``src`` preserving aspect ratio (round-to-nearest
// dims), and copies the result into the centered ROI.
//
// ``dst`` is caller-owned (the ORT Preprocessor pre-allocates it once and reuses it every
// frame); ``cv::resize`` still allocates its intermediate result per call, so only the
// destination canvas itself is allocation-free.
struct LetterboxParams {
    float scale = 1.0f;   // letterbox scale factor (min of per-axis ratios)
    float pad_x  = 0.f;   // single-side horizontal half-pad (input_size - resized_w)/2,
                          // kept as float; the ROI placement and PreprocessResult both
                          // floor it to int(dw/2).
    float pad_y  = 0.f;   // single-side vertical half-pad, same semantics as pad_x
    int   orig_w = 0;      // original image width  (pre-letterbox)
    int   orig_h = 0;      // original image height (pre-letterbox)
};

// Letterbox ``src`` into ``dst`` (input_size x input_size CV_8UC3). Writes the transform
// parameters into ``out`` so the postprocessor can invert them. Returns void; the caller
// owns ``dst``'s storage.
void letterboxInto(
    const cv::Mat& src,
    cv::Mat& dst,
    int input_size,
    LetterboxParams& out
);
