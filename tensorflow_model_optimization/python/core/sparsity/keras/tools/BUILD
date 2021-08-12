load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_strict_library(
    name = "sparsity_tooling",
    srcs = ["sparsity_tooling.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:metrics",
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_schedule",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_wrapper",
    ],
)

py_test(
    name = "sparsity_tooling_test",
    size = "medium",
    srcs = ["sparsity_tooling_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":sparsity_tooling",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/sparsity/keras:test_utils",
    ],
)

py_binary(
    name = "evaluate_pruning",
    srcs = ["evaluate_pruning.py"],
    python_version = "PY3",
    deps = [
        ":sparsity_tooling",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune",
    ],
)
