#include "benchmark.h"
#include "yolov8.h"
#include "logger.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#  include <psapi.h>
#else
#  include <fstream>
#endif

double Benchmark::rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        // WorkingSetSize is in bytes.
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    // Linux: parse VmRSS from /proc/self/status (kB -> MB).
    std::ifstream f("/proc/self/status");
    if (!f.is_open()) return 0.0;

    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            // "VmRSS:	   12345 kB"
            long kb = 0;
            if (std::sscanf(line.c_str(), "VmRSS: %ld kB", &kb) == 1) {
                return static_cast<double>(kb) / 1024.0;
            }
        }
    }
    return 0.0;
#endif
}

double Benchmark::percentile(const std::vector<double>& sorted, double q) {
    // Nearest-rank interpolation-free percentile, matching numpy's default
    if (sorted.empty()) return 0.0;
    const double clamped = std::clamp(q, 0.0, 1.0);
    const double pos = clamped * (sorted.size() - 1);
    const size_t idx = std::min(
        static_cast<size_t>(std::llround(pos)),
        sorted.size() - 1
    );
    return sorted[idx];
}

BenchmarkSummary Benchmark::run(
    YOLOv8& detector,
    const std::vector<cv::Mat>& images,
    const std::vector<std::string>& image_paths,
    int warmup_iters,
    int timed_iters,
    float conf_threshold,
    float iou_threshold,
    std::vector<BenchmarkRow>* rows_out
) {
    BenchmarkSummary summary;
    if (images.empty() || timed_iters <= 0) {
        return summary;
    }

    auto pick = [&](int i) -> std::pair<cv::Mat, std::string> {
        const size_t idx = static_cast<size_t>(i) % images.size();
        const std::string label = (idx < image_paths.size())
            ? image_paths[idx]
            : ("img_" + std::to_string(idx));
        return { images[idx], label };
    };

    // Warmup — first Run() pays for ORT graph optimizations, thread
    // pool spin-up, and any kernel JIT. Don't time it.
    for (int i = 0; i < warmup_iters; ++i) {
        auto [img, _] = pick(0);
        (void)_;  // unused
        detector.detect(img, conf_threshold, iou_threshold);
    }

    std::vector<BenchmarkRow> rows;
    rows.reserve(static_cast<size_t>(timed_iters));
    std::vector<double> total_times;
    total_times.reserve(static_cast<size_t>(timed_iters));
    double peak_rss = 0.0;

    for (int i = 0; i < timed_iters; ++i) {
        auto [img, label] = pick(i);
        auto t0 = std::chrono::high_resolution_clock::now();
        detector.detect(img, conf_threshold, iou_threshold);
        auto t1 = std::chrono::high_resolution_clock::now();

        const double total_ms = std::chrono::duration<double, std::milli>(
            t1 - t0
        ).count();
        total_times.push_back(total_ms);

        const auto& p = detector.profile();
        BenchmarkRow row;
        row.index         = i;
        row.image         = label;
        row.preprocess_ms = p.preprocess_ms;
        row.infer_ms      = p.infer_ms;
        row.postprocess_ms= p.postprocess_ms;
        row.total_ms      = total_ms;

        const double rss = rss_mb();
        row.rss_mb = rss;
        peak_rss = std::max(peak_rss, rss);

        rows.push_back(row);
    }

    if (rows_out) *rows_out = rows;

    // Aggregate.
    summary.count = static_cast<int>(total_times.size());
    summary.peak_rss_mb = peak_rss;

    double sum = 0.0;
    for (double t : total_times) sum += t;
    summary.mean_ms = sum / total_times.size();
    summary.throughput_fps = (summary.mean_ms > 0.0)
        ? 1000.0 / summary.mean_ms
        : 0.0;

    std::vector<double> sorted = total_times;
    std::sort(sorted.begin(), sorted.end());
    summary.min_ms = sorted.front();
    summary.max_ms = sorted.back();
    summary.p50_ms = percentile(sorted, 0.50);
    summary.p95_ms = percentile(sorted, 0.95);
    summary.p99_ms = percentile(sorted, 0.99);

    double sq = 0.0;
    for (double t : total_times) {
        const double d = t - summary.mean_ms;
        sq += d * d;
    }
    summary.stddev_ms = (total_times.size() > 0)
        ? std::sqrt(sq / total_times.size())
        : 0.0;

    return summary;
}

BenchmarkSummary Benchmark::summarize(const std::vector<BenchmarkRow>& rows) {
    BenchmarkSummary s;
    s.count = static_cast<int>(rows.size());
    if (rows.empty()) return s;

    std::vector<double> totals;
    totals.reserve(rows.size());
    for (const auto& r : rows) {
        totals.push_back(r.total_ms);
        s.peak_rss_mb = std::max(s.peak_rss_mb, r.rss_mb);
    }
    double sum = 0.0;
    for (double t : totals) sum += t;
    s.mean_ms = sum / totals.size();
    s.throughput_fps = (s.mean_ms > 0.0) ? 1000.0 / s.mean_ms : 0.0;

    std::sort(totals.begin(), totals.end());
    s.min_ms = totals.front();
    s.max_ms = totals.back();
    s.p50_ms = percentile(totals, 0.50);
    s.p95_ms = percentile(totals, 0.95);
    s.p99_ms = percentile(totals, 0.99);

    double sq = 0.0;
    for (double t : totals) {
        const double d = t - s.mean_ms;
        sq += d * d;
    }
    s.stddev_ms = std::sqrt(sq / totals.size());
    return s;
}

void Benchmark::write_csv(const std::string& path,
                          const std::vector<BenchmarkRow>& rows) {
    std::ofstream out(path);
    if (!out.is_open()) {
        LOG_ERROR("Failed to open CSV for writing: " + path);
        return;
    }
    out << "index,image,preprocess_ms,infer_ms,postprocess_ms,total_ms,rss_mb\n";
    out << std::fixed << std::setprecision(4);
    for (const auto& r : rows) {
        // Quote the image path to handle commas/spaces.
        out << r.index << ",\"" << r.image << "\","
            << r.preprocess_ms << ","
            << r.infer_ms << ","
            << r.postprocess_ms << ","
            << r.total_ms << ","
            << r.rss_mb << "\n";
    }
    LOG_INFO("Wrote benchmark CSV: " + path
             + " (" + std::to_string(rows.size()) + " rows)");
}

void Benchmark::write_json(const std::string& path,
                           const BenchmarkSummary& s,
                           const std::string& model_path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        LOG_ERROR("Failed to open JSON for writing: " + path);
        return;
    }
    out << std::fixed << std::setprecision(4);
    out << "{\n";
    out << "  \"model\": \"" << model_path << "\",\n";
    out << "  \"count\": " << s.count << ",\n";
    out << "  \"min_ms\": "    << s.min_ms    << ",\n";
    out << "  \"max_ms\": "    << s.max_ms    << ",\n";
    out << "  \"mean_ms\": "   << s.mean_ms   << ",\n";
    out << "  \"stddev_ms\": " << s.stddev_ms << ",\n";
    out << "  \"p50_ms\": "    << s.p50_ms    << ",\n";
    out << "  \"p95_ms\": "    << s.p95_ms    << ",\n";
    out << "  \"p99_ms\": "    << s.p99_ms    << ",\n";
    out << "  \"throughput_fps\": " << s.throughput_fps << ",\n";
    out << "  \"peak_rss_mb\": " << s.peak_rss_mb << "\n";
    out << "}\n";
    LOG_INFO("Wrote benchmark JSON: " + path);
}

void Benchmark::print_summary(const BenchmarkSummary& s, std::ostream& os) {
    os << std::fixed << std::setprecision(2);
    os << "=== Benchmark ===\n";
    os << "  count       : " << s.count          << "\n";
    os << "  mean        : " << s.mean_ms        << " ms\n";
    os << "  p50         : " << s.p50_ms         << " ms\n";
    os << "  p95         : " << s.p95_ms         << " ms\n";
    os << "  p99         : " << s.p99_ms         << " ms\n";
    os << "  min         : " << s.min_ms         << " ms\n";
    os << "  max         : " << s.max_ms         << " ms\n";
    os << "  stddev      : " << s.stddev_ms      << " ms\n";
    os << "  throughput  : " << s.throughput_fps << " FPS\n";
    os << "  peak RSS    : " << s.peak_rss_mb    << " MB\n";
}
