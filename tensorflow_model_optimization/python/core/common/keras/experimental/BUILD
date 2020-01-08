package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "pruning_algorithm",
    srcs = ["pruning_algorithm.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/common:algorithm",
        "//tensorflow_model_optimization/python/core/common/keras:wrapper",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_schedule",
    ],
)

py_test(
    name = "pruning_algorithm_test",
    size = "medium",
    srcs = ["pruning_algorithm_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":pruning_algorithm",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/common:algorithm",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_schedule",
    ],
)
