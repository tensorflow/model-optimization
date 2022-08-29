load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_binary")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
)

py_strict_binary(
    name = "mnist_cnn",
    srcs = [
        "mnist_cnn.py",
    ],
    python_version = "PY3",
    deps = [
        # absl:app dep1,
        # absl/flags dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
        "//tensorflow_model_optimization/python/core/clustering/keras:clustering_callbacks",
    ],
)
