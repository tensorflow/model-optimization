package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "tflite",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [],
)

py_library(
    name = "tflite_quantizers",
    srcs = [
        "tflite_quantizers.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
    ],
)

py_test(
    name = "tflite_quantizers_test",
    srcs = [
        "tflite_quantizers_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_quantizers",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "tflite_quantize_configs",
    srcs = [
        "tflite_quantize_configs.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_config",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
    ],
)

py_library(
    name = "tflite_quantize_registry",
    srcs = [
        "tflite_quantize_registry.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_config",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/layers:conv_batchnorm",
        "//tensorflow_model_optimization/python/core/quantization/keras/tflite:tflite_quantize_configs",
        "//tensorflow_model_optimization/python/core/quantization/keras/tflite:tflite_quantizers",
    ],
)

py_test(
    name = "tflite_quantize_registry_test",
    srcs = [
        "tflite_quantize_registry_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_quantize_registry",
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_library(
    name = "tflite_transforms",
    srcs = [
        "tflite_transforms.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_aware_activation",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_layer",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:transforms",
        "//tensorflow_model_optimization/python/core/quantization/keras/layers:conv_batchnorm",
        "//tensorflow_model_optimization/python/core/quantization/keras/tflite:tflite_quantize_configs",
    ],
)

py_test(
    name = "tflite_transforms_test",
    srcs = [
        "tflite_transforms_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_transforms",
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:model_transformer",
        "//tensorflow_model_optimization/python/core/quantization/keras/layers:conv_batchnorm_test_utils",
    ],
)

py_library(
    name = "tflite_quantize_layout_transform",
    srcs = [
        "tflite_quantize_layout_transform.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_transforms",
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_layout_transform",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:model_transformer",
    ],
)
