load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library", "py_strict_test")

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_strict_library(
    name = "keras",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":prunable_layer",  # buildcleaner: keep
        ":prune",  # buildcleaner: keep
        ":pruning_callbacks",  # buildcleaner: keep
        ":pruning_policy",  # buildcleaner: keep
        ":pruning_schedule",  # buildcleaner: keep
    ],
)

py_strict_library(
    name = "prune",
    srcs = ["prune.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_schedule",
        ":pruning_wrapper",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:metrics",
    ],
)

py_strict_library(
    name = "prunable_layer",
    srcs = ["prunable_layer.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
    ],
)

py_strict_library(
    name = "prune_registry",
    srcs = ["prune_registry.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":prunable_layer",
        # tensorflow dep1,
        # python/keras:base_layer tensorflow dep2,
    ],
)

py_strict_library(
    name = "pruning_policy",
    srcs = ["pruning_policy.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_wrapper",
        # tensorflow dep1,
    ],
)

py_strict_library(
    name = "pruning_schedule",
    srcs = ["pruning_schedule.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
        # tensorflow dep1,
    ],
)

py_strict_test(
    name = "pruning_schedule_test",
    size = "medium",
    srcs = ["pruning_schedule_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_schedule",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
        # python/keras tensorflow dep2,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_library(
    name = "pruning_wrapper",
    srcs = ["pruning_wrapper.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":prunable_layer",
        ":prune_registry",
        ":pruning_impl",
        ":pruning_schedule",
        ":pruning_utils",
        # numpy dep1,
        # tensorflow:tensorflow_no_contrib dep1,
        # python/keras/utils:generic_utils tensorflow dep2,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras:metrics",
        "//tensorflow_model_optimization/python/core/keras:utils",
    ],
)

py_strict_library(
    name = "pruning_callbacks",
    srcs = ["pruning_callbacks.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_wrapper",
        # numpy dep1,
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_test(
    name = "pruning_callbacks_test",
    size = "medium",
    srcs = ["pruning_callbacks_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":prune",
        ":pruning_callbacks",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        # python/keras tensorflow dep2,
        "//tensorflow_model_optimization/python/core/keras:test_utils",
    ],
)

py_strict_library(
    name = "pruning_impl",
    srcs = ["pruning_impl.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_utils",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_library(
    name = "pruning_utils",
    srcs = ["pruning_utils.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_strict_library(
    name = "estimator_utils",
    srcs = ["estimator_utils.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_wrapper",
        # tensorflow dep1,
        # tensorflow:tensorflow_compat_v1_estimator dep1,
        # python/framework:for_generated_wrappers tensorflow dep2,
    ],
)

py_strict_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":prune",
        ":pruning_utils",
        ":pruning_wrapper",
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_strict_test(
    name = "prune_test",
    size = "medium",
    srcs = ["prune_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":prunable_layer",
        ":prune",
        ":pruning_schedule",
        ":pruning_wrapper",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        # python/keras tensorflow dep2,
        "//tensorflow_model_optimization/python/core/keras:test_utils",
    ],
)

py_strict_test(
    name = "prune_integration_test",
    size = "medium",
    timeout = "long",
    srcs = ["prune_integration_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":prune",
        ":prune_registry",
        ":pruning_callbacks",
        ":pruning_schedule",
        ":pruning_wrapper",
        ":test_utils",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        # python/keras tensorflow dep2,
        "//tensorflow_model_optimization/python/core/keras:test_utils",
    ],
)

py_strict_test(
    name = "prune_distributed_test",
    size = "medium",
    timeout = "long",
    srcs = ["prune_distributed_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":prune",
        ":pruning_callbacks",
        ":pruning_schedule",
        ":test_utils",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:test_utils",
    ],
)

py_strict_test(
    name = "prune_registry_test",
    size = "medium",
    srcs = ["prune_registry_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":prunable_layer",
        ":prune_registry",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
    ],
)

py_strict_test(
    name = "pruning_wrapper_test",
    size = "medium",
    srcs = ["pruning_wrapper_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_schedule",
        ":pruning_wrapper",
        # tensorflow dep1,
    ],
)

py_strict_test(
    name = "pruning_impl_test",
    size = "medium",
    srcs = ["pruning_impl_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_impl",
        ":pruning_schedule",
        ":pruning_utils",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        # python/keras tensorflow dep2,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_test(
    name = "pruning_utils_test",
    size = "medium",
    srcs = ["pruning_utils_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_utils",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_test(
    name = "pruning_policy_test",
    size = "medium",
    srcs = ["pruning_policy_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":prune",
        ":pruning_policy",
        ":pruning_schedule",
        ":pruning_wrapper",
        # tensorflow dep1,
    ],
)
