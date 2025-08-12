#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace lsbm {

std::vector<double> extract_all_features(const cv::Mat &image);      // 135 fitur
std::vector<double> extract_selected_features(const cv::Mat &image); // 54 fitur

// helper: allow python to pass filename or ndarray via bindings
}

