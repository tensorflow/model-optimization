package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
    "//third_party/tensorflow:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "quant_ops",
    srcs = ["quant_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        # python:array_ops tensorflow dep2,
        # python:framework_ops tensorflow dep2,
        # python:init_ops tensorflow dep2,
        # python:math_ops tensorflow dep2,
        # python:state_ops tensorflow dep2,
        # python:training tensorflow dep2,
        # python:variable_scope tensorflow dep2,
    ],
)

py_test(
    name = "quant_ops_test",
    size = "small",
    srcs = ["quant_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":quant_ops",
        # python:array_ops tensorflow dep2,
        # python:dtypes tensorflow dep2,
        # python:framework_ops tensorflow dep2,
        # python:partitioned_variables tensorflow dep2,
        # python:platform_test tensorflow dep2,
        # python:session tensorflow dep2,
        # python:variable_scope tensorflow dep2,
        # python:variables tensorflow dep2,
    ],
)

py_library(
    name = "quantize_emulatable_layer",
    srcs = [
        "quantize_emulatable_layer.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
        # tensorflow dep1,
        # python/keras tensorflow dep2,
    ],
)

py_library(
    name = "quantize_emulate_registry",
    srcs = [
        "quantize_emulate_registry.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
        # tensorflow dep1,
        # python/keras tensorflow dep2,
    ],
)

py_library(
    name = "quantize_emulate_wrapper",
    srcs = [
        "quantize_emulate_wrapper.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quant_ops",
        ":quantize_emulatable_layer",
        ":quantize_emulate_registry",
        # six dep1,
        # tensorflow dep1,
        # python/keras tensorflow dep2,
    ],
)

py_library(
    name = "quantize_emulate",
    srcs = [
        "quantize_emulate.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_annotate",
        ":quantize_aware_activation",
        ":quantize_emulate_wrapper",
        # tensorflow dep1,
        # python/keras tensorflow dep2,
    ],
)

py_library(
    name = "quantize_annotate",
    srcs = [
        "quantize_annotate.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_emulatable_layer",
        ":quantize_emulate_registry",
        # tensorflow dep1,
        # python/keras tensorflow dep2,
    ],
)

py_test(
    name = "quantize_annotate_test",
    srcs = [
        "quantize_annotate_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_annotate",
        ":quantize_emulatable_layer",
        # tensorflow dep1,
        # python/keras tensorflow dep2,
    ],
)

py_library(
    name = "quantize_aware_activation",
    srcs = [
        "quantize_aware_activation.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quant_ops",
        # tensorflow dep1,
        # python/keras tensorflow dep2,
    ],
)

py_test(
    name = "quantize_aware_activation_test",
    srcs = [
        "quantize_aware_activation_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_aware_activation",
        # tensorflow dep1,
        # python/keras tensorflow dep2,
    ],
)

py_library(
    name = "utils",
    srcs = [
        "utils.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_emulate_wrapper",
        # tensorflow dep1,
    ],
)

py_test(
    name = "quantize_emulate_test",
    srcs = ["quantize_emulate_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_aware_activation",
        ":quantize_emulate",
        ":quantize_emulate_wrapper",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
    ],
)

py_test(
    name = "quantize_emulate_integration_test",
    srcs = ["quantize_emulate_integration_test.py"],
    deps = [
        ":quantize_emulate",
        ":quantize_emulate_wrapper",
        ":utils",
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:test_utils",
    ],
)
