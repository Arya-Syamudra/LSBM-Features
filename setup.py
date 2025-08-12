from setuptools import setup, Extension
import sys
import pybind11
import os
from setuptools.command.build_ext import build_ext

opencv_includes = []  # let compiler find OpenCV via pkg-config
opencv_libs = []

ext_modules = [
    Extension(
        "lsbm_features",
        sources=["src/bindings.cpp", "src/lsbm_features.cpp"],
        include_dirs=[pybind11.get_include(), "include"],
        language="c++",
        extra_compile_args=["-std=c++17"]
    )
]

setup(
    name="lsbm_features",
    version="0.1.0",
    description="LSBM features (Liu 2006) - C++ implementation",
    ext_modules=ext_modules,
    install_requires=["pybind11", "opencv-python"],
)

