load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_strict_library(
    name = "clustering",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_model_optimization/python/core/clustering/keras",  # buildcleaner: keep
    ],
)
