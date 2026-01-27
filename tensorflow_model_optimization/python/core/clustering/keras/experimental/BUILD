# Placeholder: load py_library

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "experimental",
    srcs = [
        "__init__.py",
    ],
    strict_deps = True,
    deps = [
        ":cluster",  # buildcleaner: keep
    ],
)

py_library(
    name = "cluster",
    srcs = ["cluster.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
    ],
)
