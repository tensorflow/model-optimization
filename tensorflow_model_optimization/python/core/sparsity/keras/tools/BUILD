# Placeholder: load py_binary
# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_applicable_licenses = ["//tensorflow_model_optimization:license"],
    default_visibility = [
        "//tensorflow_model_optimization:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "sparsity_tooling",
    srcs = ["sparsity_tooling.py"],
    # strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
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
    # strict_deps = False,
    visibility = ["//visibility:public"],
    deps = [
        ":sparsity_tooling",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/sparsity/keras:test_utils",
    ],
)

py_library(
    name = "evaluate_pruning_lib",
    srcs = ["evaluate_pruning.py"],
    deps = [
        ":sparsity_tooling",
        # absl:app dep1,
        # absl/flags dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune",
    ],
)

py_binary(
    name = "evaluate_pruning",
    srcs = ["evaluate_pruning.py"],
    # strict_deps = False,
    deps = [
        ":evaluate_pruning_lib",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
    ],
)

py_test(
    name = "evaluate_pruning_test",
    size = "medium",
    srcs = ["evaluate_pruning_test.py"],
    deps = [
        ":evaluate_pruning_lib",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_library(
    name = "check_sparsity_m_by_n",
    srcs = ["check_sparsity_m_by_n.py"],
    # strict_deps = True,
    deps = [
        # absl:app dep1,
        # absl/flags dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_utils",
    ],
)
