#include "yolov8.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "logger.h"

void YOLOv8::initializeSession(const std::string& model_path,
                               int intra_op_threads, int inter_op_threads) {
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );
    session_options_.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    // Thread counts: 0 leaves the ORT default (intra-op = all cores); a
    // positive value pins the pool (--intra-op-threads / --inter-op-threads
    // on the CLI). With ORT_PARALLEL, inter_op must be set explicitly or
    // ORT keeps its per-branch default pool.
    if (intra_op_threads > 0) {
        session_options_.SetIntraOpNumThreads(intra_op_threads);
    }
    if (inter_op_threads > 0) {
        session_options_.SetInterOpNumThreads(inter_op_threads);
    }

    session_ = Ort::Session(env_, model_path.c_str(), session_options_);
}

YOLOv8::YOLOv8(
    const std::string& model_path,
    const std::vector<std::string>& class_names,
    int input_size,
    int intra_op_threads,
    int inter_op_threads
)
    :
    env_(ORT_LOGGING_LEVEL_WARNING, "YOLOv8"),
    input_size_(input_size),
    preprocessor_(
        input_size,
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
    ),
    postprocessor_(),
    class_names_(class_names)
{
    initializeSession(model_path, intra_op_threads, inter_op_threads);

    size_t input_count = session_.GetInputCount();
    input_names_.resize(input_count);
    input_names_ptr_.resize(input_count);
    for (size_t i = 0; i < input_count; ++i) {
        auto name = session_.GetInputNameAllocated(i, allocator_);
        input_names_[i] = name.get();
        input_names_ptr_[i] = input_names_[i].c_str();
    }

    size_t output_count = session_.GetOutputCount();
    output_names_.resize(output_count);
    output_names_ptr_.resize(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        auto name = session_.GetOutputNameAllocated(i, allocator_);
        output_names_[i] = name.get();
        output_names_ptr_[i] = output_names_[i].c_str();
    }

    LOG_INFO("YOLOv8 initialized (input_size="
        + std::to_string(input_size)
        + ", classes=" + std::to_string(class_names.size()) + ")");
}

YOLOv8::~YOLOv8() = default;

std::vector<Detection> YOLOv8::detect(
    const cv::Mat& image,
    float conf_threshold,
    float iou_threshold
) {
    using clock = std::chrono::high_resolution_clock;
    auto t_total_start = clock::now();

    // 1) Preprocess — letterbox + HWC->CHW into the pre-allocated tensor.
    auto t1 = clock::now();
    PreprocessResult pre;
    Ort::Value& input_tensor = preprocessor_.process(image, pre);
    auto t2 = clock::now();

    // 2) Infer. `input_tensor` is a reference into the preprocessor's
    //    storage; it must outlive the Run() call.
    auto output_tensors = session_.Run(
        run_options_,
        input_names_ptr_.data(),
        &input_tensor,
        1,
        output_names_ptr_.data(),
        output_names_ptr_.size()
    );
    auto t3 = clock::now();

    if (output_tensors.empty()) {
        throw std::runtime_error("ORT session produced no outputs");
    }

    // 3) Postprocess — decode 8400 anchors, class-wise NMS, scale back.
    auto detections = postprocessor_.process(
        output_tensors[0], image, pre, class_names_,
        conf_threshold, iou_threshold
    );
    auto t4 = clock::now();

    profile_result_.preprocess_ms  = std::chrono::duration<double, std::milli>(t2 - t1).count();
    profile_result_.infer_ms       = std::chrono::duration<double, std::milli>(t3 - t2).count();
    profile_result_.postprocess_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    profile_result_.total_ms       = std::chrono::duration<double, std::milli>(t4 - t_total_start).count();

    return detections;
}

const ProfileResult& YOLOv8::profile() const {
    return profile_result_;
}
