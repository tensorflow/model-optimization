load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_strict_library(
    name = "tensor_encoding",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/encoders",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/stages",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils",
    ],
)
