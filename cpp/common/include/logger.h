#pragma once

#include <iostream>
#include <chrono>
#include <iomanip>

class Logger {
public:
    enum class Level {
        INFO,
        WARNING,
        ERROR
    };

    static void log(Level level, const std::string& msg)
    {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;

#ifdef _WIN32
        localtime_s(&tm_buf, &time);
#else
        localtime_r(&time, &tm_buf);
#endif

        std::string prefix;
        switch (level) {
            case Level::INFO:
                prefix = "[INFO]";
                break;
            case Level::WARNING:
                prefix = "[WARN]";
                break;
            case Level::ERROR:
                prefix = "[ERROR]";
                break;
        }

        std::cout
            << prefix
            << " "
            << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
            << " "
            << msg
            << std::endl;
    }
};

#define LOG_INFO(msg) Logger::log(Logger::Level::INFO, msg)
#define LOG_WARN(msg) Logger::log(Logger::Level::WARNING, msg)
#define LOG_ERROR(msg) Logger::log(Logger::Level::ERROR, msg)
