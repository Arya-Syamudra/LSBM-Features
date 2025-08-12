cmake_minimum_required(VERSION 3.10)
project(lsbm_features)

set(CMAKE_CXX_STANDARD 17)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)

pybind11_add_module(lsbm_features src/bindings.cpp src/lsbm_features.cpp)
target_include_directories(lsbm_features PRIVATE include)
target_link_libraries(lsbm_features PRIVATE ${OpenCV_LIBS})

