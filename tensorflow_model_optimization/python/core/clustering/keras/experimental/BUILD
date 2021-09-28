load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_strict_library(
    name = "experimental",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":cluster",  # buildcleaner: keep
    ],
)

py_strict_library(
    name = "cluster",
    srcs = ["cluster.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
    ],
)
