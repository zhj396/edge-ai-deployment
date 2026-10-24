#include "draw_utils.h"

#include <cstdio>

void drawDetections(cv::Mat& image, const std::vector<Detection>& detections)
{
    for (const auto& det : detections)
    {
        cv::rectangle(
            image,
            det.box,
            cv::Scalar(0, 255, 0),
            2
        );

        // Render the confidence at 2 decimal places on the label.
        char conf_buf[16];
        std::snprintf(conf_buf, sizeof(conf_buf), "%.2f", det.conf);
        std::string label = det.class_name + " " + conf_buf;

        int baseline = 0;

        cv::Size text_size =
            cv::getTextSize(
                label,
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
                &baseline
            );

        int x = det.box.x;
        int y = det.box.y;

        cv::rectangle(
            image,
            cv::Point(x, y - text_size.height - 10),
            cv::Point(x + text_size.width, y),
            cv::Scalar(0, 255, 0),
            cv::FILLED
        );

        cv::putText(
            image,
            label,
            cv::Point(x, y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 0, 0),
            2
        );
    }
}
