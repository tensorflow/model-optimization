load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

package(default_visibility = ["//tensorflow_model_optimization:__subpackages__"])

licenses(["notice"])

py_strict_library(
    name = "quantization",
    srcs = ["__init__.py"],
    deps = [
        "//tensorflow_model_optimization/python/core/quantization/keras",  # buildcleaner: keep
    ],
)
