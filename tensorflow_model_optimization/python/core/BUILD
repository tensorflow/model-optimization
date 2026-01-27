# Placeholder: load py_library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "core",
    srcs = ["__init__.py"],
    strict_deps = True,
    deps = [
        "//tensorflow_model_optimization/python/core/api",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core/clustering",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core/internal",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core/keras",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core/quantization",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core/sparsity",  # buildcleaner: keep
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
    strict_deps = True,
)
