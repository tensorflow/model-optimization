package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
)

py_binary(
    name = "mnist_estimator",
    srcs = [
        "dataset.py",
        "mnist_estimator.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # absl/flags dep1,
        # numpy dep1,
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/sparsity/keras:estimator_utils",
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune",
        "//third_party/tensorflow_models/official/utils",
    ],
)

py_binary(
    name = "mnist_cnn",
    srcs = [
        "mnist_cnn.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # absl:app dep1,
        # absl/flags dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_callbacks",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_schedule",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_wrapper",
    ],
)
