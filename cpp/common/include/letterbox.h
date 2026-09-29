#pragma once

#include <opencv2/opencv.hpp>

#include "types.h"

// Backend-agnostic letterbox — the C++ mirror of ``src/preprocess.py::letterbox``.
//
// Fills ``dst`` (a pre-allocated ``input_size x input_size`` ``CV_8UC3`` buffer owned by the
// caller) with YOLO gray (114,114,114), resizes ``src`` preserving aspect ratio, and copies
// the result into the centered ROI. The resized dimensions are **rounded**
// (``cv::saturate_cast<int>`` ≡ ``cvRound``) to match the Python path's
// ``int(round(shape * r))``, so both backends letterbox to identical sizes for
// non-integer scales.
//
// ``dst`` is caller-owned (the ORT Preprocessor pre-allocates it once and reuses it every
// frame) so this stays zero-allocation per call.
struct LetterboxParams {
    float scale = 1.0f;   // letterbox scale factor (min of per-axis ratios)
    int   pad_x  = 0;     // single-side horizontal padding (left/top corner of the ROI)
    int   pad_y  = 0;      // single-side vertical padding
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
