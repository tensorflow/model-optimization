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
    deps = [
        # absl:app dep1,
        # absl/flags dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_callbacks",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_schedule",
    ],
)

py_strict_binary(
    name = "mnist_e2e",
    srcs = [
        "mnist_e2e.py",
    ],
    deps = [
        # absl:app dep1,
        # absl/flags dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras:test_utils",
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_callbacks",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_schedule",
    ],
)

py_strict_binary(
    name = "mnist_e2e_sparsity2x4",
    srcs = ["mnist_e2e_sparsity2x4.py"],
    deps = [
        # absl:app dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras:test_utils",
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_callbacks",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_schedule",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_utils",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_wrapper",
    ],
)
