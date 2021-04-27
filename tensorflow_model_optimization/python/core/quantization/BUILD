load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

package(default_visibility = ["//tensorflow_model_optimization:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_strict_library(
    name = "quantization",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_model_optimization/python/core/quantization/keras",  # buildcleaner: keep
    ],
)
