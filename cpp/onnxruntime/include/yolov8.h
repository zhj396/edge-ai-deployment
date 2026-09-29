#pragma once

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "types.h"
#include "preprocessor.h"
#include "postprocessor.h"

// YOLOv8 inference orchestrator. Owns the ORT session and two pipeline
// stages: Preprocessor (letterbox + HWC->CHW) and Postprocessor (NMS +
// scale-back). The detect() call measures per-stage wall-clock time and
// exposes the breakdown via profile().
//
// CPU-only deployment target. CUDA EP / TensorRT EP are intentionally
// out of scope for the current release; reintroduce when an actual
// GPU build target is wired up.
class YOLOv8 {
public:
    YOLOv8(
        const std::string& model_path,
        const std::vector<std::string>& class_names,
        int input_size = 640,
        int intra_op_threads = 0,   // 0 = ORT default (all cores)
        int inter_op_threads = 0    // 0 = ORT default
    );

    ~YOLOv8();

    std::vector<Detection> detect(
        const cv::Mat& image,
        float conf_threshold,
        float iou_threshold
    );

    const ProfileResult& profile() const;

private:
    void initializeSession(const std::string& model_path,
                           int intra_op_threads, int inter_op_threads);

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::RunOptions run_options_;

    int input_size_;

    // Pipeline stages. Order of declaration matters: preprocessor_ is
    // initialized with the ORT memory-info, and postprocessor_ is default.
    Preprocessor  preprocessor_;
    Postprocessor postprocessor_;

    ProfileResult profile_result_;

    std::vector<std::string> class_names_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_ptr_;
    std::vector<const char*> output_names_ptr_;
};
