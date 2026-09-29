#pragma once

#include <string>
#include <vector>
#include <iosfwd>
#include <opencv2/opencv.hpp>

class YOLOv8;  // forward decl — full include in benchmark.cpp

// One row of per-iteration benchmark results. `rss_mb` is sampled
// AFTER the iteration completes (i.e., close to the peak for that
// iter). Use BenchmarkSummary::peak_rss_mb for the global maximum.
struct BenchmarkRow {
    int    index = 0;
    std::string image;
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
    double rss_mb = 0.0;
};

// Aggregate stats over all timed iterations.
struct BenchmarkSummary {
    int    count = 0;
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double stddev_ms = 0.0;
    double peak_rss_mb = 0.0;
    double throughput_fps = 0.0;  // 1000 / mean_ms
};

class Benchmark {
public:
    // Run `warmup_iters` warm-up iterations (not timed) followed by
    // `timed_iters` timed iterations over the image set. The timed loop
    // cycles through `images`; `images` must be non-empty (an empty set
    // returns an empty summary). Pass `rows_out`
    // to capture per-iteration data; pass nullptr to skip.
    static BenchmarkSummary run(
        YOLOv8& detector,
        const std::vector<cv::Mat>& images,
        const std::vector<std::string>& image_paths,
        int warmup_iters,
        int timed_iters,
        float conf_threshold,
        float iou_threshold,
        std::vector<BenchmarkRow>* rows_out = nullptr
    );

    // Aggregate an existing vector of rows into a summary. Used by
    // directory mode where each image is timed once (no warmup loop).
    static BenchmarkSummary summarize(const std::vector<BenchmarkRow>& rows);

    // Resident Set Size in MB.
    //   - Linux:  /proc/self/status  ->  VmRSS
    //   - Windows: GetProcessMemoryInfo -> WorkingSetSize
    static double rss_mb();

    // Persist per-iteration data to CSV (header row included).
    static void write_csv(const std::string& path,
                          const std::vector<BenchmarkRow>& rows);

    // Persist summary to JSON. `model_path` is echoed into the JSON
    // for traceability. Pretty-printed with 2-space indent.
    static void write_json(const std::string& path,
                           const BenchmarkSummary& summary,
                           const std::string& model_path);

    // Pretty-print summary to a stream.
    static void print_summary(const BenchmarkSummary& s, std::ostream& os);

private:
    static double percentile(const std::vector<double>& sorted, double q);
};
