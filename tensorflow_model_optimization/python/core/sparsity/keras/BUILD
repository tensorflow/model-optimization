package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":prunable_layer",
        ":prune",
        ":pruning_callbacks",
        ":pruning_schedule",
    ],
)

py_library(
    name = "prune",
    srcs = ["prune.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_schedule",
        ":pruning_wrapper",
        # python:util tensorflow dep2,
        # python/keras tensorflow dep2,
        # python/keras:engine tensorflow dep2,
        # python/keras:generic_utils tensorflow dep2,
    ],
)

py_library(
    name = "prunable_layer",
    srcs = ["prunable_layer.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
    ],
)

py_library(
    name = "prune_registry",
    srcs = ["prune_registry.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":prunable_layer",
        # python/keras:layers_base tensorflow dep2,
    ],
)

py_library(
    name = "pruning_schedule",
    srcs = ["pruning_schedule.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
        # python:constant_op tensorflow dep2,
        # python:dtypes tensorflow dep2,
        # python:framework_ops tensorflow dep2,
        # python:math_ops tensorflow dep2,
    ],
)

py_test(
    name = "pruning_schedule_test",
    size = "medium",
    srcs = ["pruning_schedule_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_schedule",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "pruning_wrapper",
    srcs = ["pruning_wrapper.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":prunable_layer",
        ":prune_registry",
        ":pruning_impl",
        ":pruning_schedule",
        # numpy dep1,
        # python:check_ops tensorflow dep2,
        # python:control_flow_ops tensorflow dep2,
        # python:dtypes tensorflow dep2,
        # python:framework_ops tensorflow dep2,
        # python:variables tensorflow dep2,
        # python/keras:backend tensorflow dep2,
        # python/keras:engine tensorflow dep2,
        # python/keras:generic_utils tensorflow dep2,
        # python/keras:initializers tensorflow dep2,
        # python/keras:layers_base tensorflow dep2,
        # python/keras:tf_utils tensorflow dep2,
    ],
)

py_library(
    name = "pruning_callbacks",
    srcs = ["pruning_callbacks.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_wrapper",
        # numpy dep1,
        # tensorflow dep1,
        # python:math_ops tensorflow dep2,
        # python/keras:backend tensorflow dep2,
        # python/keras:callbacks tensorflow dep2,
    ],
)

py_test(
    name = "pruning_callbacks_test",
    size = "medium",
    srcs = ["pruning_callbacks_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":prune",
        ":pruning_callbacks",
        ":pruning_schedule",
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:test_utils",
    ],
)

py_library(
    name = "pruning_impl",
    srcs = ["pruning_impl.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_utils",
        # tensorflow dep1,
        # python:array_ops tensorflow dep2,
        # python:control_flow_ops tensorflow dep2,
        # python:dtypes tensorflow dep2,
        # python:framework_ops tensorflow dep2,
        # python:math_ops tensorflow dep2,
        # python:nn_ops tensorflow dep2,
        # python:state_ops tensorflow dep2,
        # python:summary tensorflow dep2,
        # python:variables tensorflow dep2,
    ],
)

py_library(
    name = "pruning_utils",
    srcs = ["pruning_utils.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # numpy dep1,
        # python:array_ops tensorflow dep2,
        # python:constant_op tensorflow dep2,
        # python:control_flow_ops tensorflow dep2,
        # python:dtypes tensorflow dep2,
        # python:framework_ops tensorflow dep2,
        # python:init_ops tensorflow dep2,
        # python:nn_ops tensorflow dep2,
        # python:state_ops tensorflow dep2,
        # python:variable_scope tensorflow dep2,
    ],
)

py_library(
    name = "estimator_utils",
    srcs = ["estimator_utils.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_wrapper",
        # tensorflow dep1,
    ],
)

py_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":prune",
        ":pruning_wrapper",
        # numpy dep1,
        # python/keras tensorflow dep2,
        # python/keras:backend tensorflow dep2,
        # python/keras:saving tensorflow dep2,
    ],
)

py_test(
    name = "prune_test",
    size = "medium",
    srcs = ["prune_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":prune",
        ":pruning_callbacks",
        ":pruning_schedule",
        ":test_utils",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_test(
    name = "prune_integration_test",
    size = "medium",
    timeout = "long",
    srcs = ["prune_integration_test.py"],
    srcs_version = "PY2AND3",
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

py_test(
    name = "prune_distributed_test",
    size = "medium",
    timeout = "long",
    srcs = ["prune_distributed_test.py"],
    srcs_version = "PY2AND3",
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

py_test(
    name = "prune_registry_test",
    size = "medium",
    srcs = ["prune_registry_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":prunable_layer",
        ":prune_registry",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
    ],
)

py_test(
    name = "pruning_wrapper_test",
    size = "medium",
    srcs = ["pruning_wrapper_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_schedule",
        ":pruning_wrapper",
    ],
)

py_test(
    name = "pruning_impl_test",
    size = "medium",
    srcs = ["pruning_impl_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_impl",
        ":pruning_schedule",
        # numpy dep1,
    ],
)

py_test(
    name = "pruning_utils_test",
    size = "medium",
    srcs = ["pruning_utils_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_utils",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
    ],
)
