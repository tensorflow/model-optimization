package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

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
    srcs_version = "PY2AND3",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # absl:app dep1,
        # absl/flags dep1,
        # python/keras tensorflow dep2,
        # python/keras/optimizer_v2 tensorflow dep2,
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_callbacks",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_schedule",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_wrapper",
    ],
)
