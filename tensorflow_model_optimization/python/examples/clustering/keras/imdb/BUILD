load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")
# Placeholder: load py_binary

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
    deps = [
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
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
    deps = [
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
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
    deps = [
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
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
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_binary(
    name = "imdb_multiple_cells",
    srcs = [
        "imdb_multiple_cells.py",
    ],
    deps = [
        ":imdb_utils",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)
