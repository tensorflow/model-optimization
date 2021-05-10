load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

package(default_visibility = ["//tensorflow_model_optimization:__subpackages__"])

licenses(["notice"])

py_strict_library(
    name = "collaborative_optimizations",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_model_optimization/python/core/quantization/keras/collaborative_optimizations/cluster_preserve",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core/quantization/keras/collaborative_optimizations/prune_preserve",  # buildcleaner: keep
    ],
)
