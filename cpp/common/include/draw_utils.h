#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#include "types.h"

// Draw final detections onto an image. Shared across C++ backends — depends only on
// ``Detection`` (types.h), NOT on any backend session header, so it lives in common/.
void drawDetections(cv::Mat& image, const std::vector<Detection>& detections);
