package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

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
        # python/keras tensorflow dep2,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_provider",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
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
        # tensorflow dep1,
        # python/keras tensorflow dep2,
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
        # python/keras tensorflow dep2,
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:transforms",
        "//tensorflow_model_optimization/python/core/quantization/keras/layers:conv_batchnorm",
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
        # python/keras tensorflow dep2,
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
        # python/keras tensorflow dep2,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_layout_transform",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:model_transformer",
    ],
)
