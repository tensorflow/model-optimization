load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
)

py_binary(
    name = "imdb_lstm",
    srcs = [
        "imdb_lstm.py",
    ],
    python_version = "PY3",
    deps = [
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
    ],
)

py_binary(
    name = "imdb_rnn",
    srcs = [
        "imdb_rnn.py",
    ],
    python_version = "PY3",
    deps = [
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
    ],
)

py_binary(
    name = "imdb_gru",
    srcs = [
        "imdb_gru.py",
    ],
    python_version = "PY3",
    deps = [
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
    ],
)

py_strict_library(
    name = "imdb_utils",
    srcs = [
        "imdb_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
    ],
)

py_binary(
    name = "imdb_multiple_cells",
    srcs = [
        "imdb_multiple_cells.py",
    ],
    python_version = "PY3",
    deps = [
        ":imdb_utils",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
    ],
)
