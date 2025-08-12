import sys
import sysconfig
from setuptools import setup, Extension
import pybind11
import os

# Lokasi header OpenCV di Ubuntu / Colab
opencv_include = "/usr/include/opencv4"

# Library OpenCV yang dibutuhkan
opencv_libs = ["opencv_core", "opencv_imgcodecs", "opencv_imgproc"]

ext_modules = [
    Extension(
        "lsbm_features",
        sources=["src/bindings.cpp", "src/lsbm_features.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            opencv_include,
            "include"
        ],
        libraries=opencv_libs,
        language="c++",
        extra_compile_args=["-std=c++17"]
    )
]

setup(
    name="lsbm_features",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="LSB Matching Steganalysis RGB Feature Extractor (Liu 2006)",
    ext_modules=ext_modules,
    install_requires=["pybind11", "opencv-python"],
    zip_safe=False,
)
