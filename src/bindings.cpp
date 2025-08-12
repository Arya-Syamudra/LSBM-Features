// src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "lsbm_features.hpp"

namespace py = pybind11;

static cv::Mat numpy_uint8_to_mat(py::array_t<uint8_t> array) {
    py::buffer_info info = array.request();
    if (info.ndim == 2) {
        return cv::Mat(info.shape[0], info.shape[1], CV_8UC1, (void*)info.ptr).clone();
    } else if (info.ndim == 3 && info.shape[2] == 3) {
        cv::Mat m(info.shape[0], info.shape[1], CV_8UC3, (void*)info.ptr);
        cv::Mat m2;
        cvtColor(m, m2, cv::COLOR_BGR2RGB); // if ndarray in BGR, try to normalize; assume BGR input
        return m2.clone();
    } else {
        throw std::runtime_error("Unsupported numpy array format for image");
    }
}

PYBIND11_MODULE(lsbm_features, m) {
    m.doc() = "LSB matching feature extractor (Liu 2006) - C++ implementation with pybind11";

    m.def("all_features", [](py::object img_obj) {
        // accept either path (str) or numpy array
        cv::Mat img;
        if (py::isinstance<py::str>(img_obj)) {
            std::string path = img_obj.cast<std::string>();
            cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
            if (bgr.empty()) throw std::runtime_error("Failed to load image: " + path);
            cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
            img = rgb;
        } else {
            // numpy array
            py::array array = img_obj.cast<py::array>();
            img = numpy_uint8_to_mat(array);
        }
        return lsbm::extract_all_features(img);
    }, "Compute all 135 features from an RGB image (accepts filepath or numpy uint8 array)");

    m.def("f54_features", [](py::object img_obj) {
        cv::Mat img;
        if (py::isinstance<py::str>(img_obj)) {
            std::string path = img_obj.cast<std::string>();
            cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
            if (bgr.empty()) throw std::runtime_error("Failed to load image: " + path);
            cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
            img = rgb;
        } else {
            py::array array = img_obj.cast<py::array>();
            img = numpy_uint8_to_mat(array);
        }
        return lsbm::extract_selected_features(img);
    }, "Compute selected 54 LSBM features (Liu 2006)");
}

