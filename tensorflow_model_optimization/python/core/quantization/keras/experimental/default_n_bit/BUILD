load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library", "py_strict_test")

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_strict_library(
    name = "default_n_bit",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [],
)

py_strict_library(
    name = "default_n_bit_quantizers",
    srcs = [
        "default_n_bit_quantizers.py",
    ],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
    ],
)

py_test(
    name = "default_n_bit_quantizers_test",
    srcs = [
        "default_n_bit_quantizers_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":default_n_bit_quantizers",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
    ],
)

py_strict_library(
    name = "default_n_bit_quantize_configs",
    srcs = [
        "default_n_bit_quantize_configs.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_config",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
    ],
)

py_strict_library(
    name = "default_n_bit_quantize_registry",
    srcs = [
        "default_n_bit_quantize_registry.py",
    ],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_config",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/experimental/default_n_bit:default_n_bit_quantize_configs",
        "//tensorflow_model_optimization/python/core/quantization/keras/experimental/default_n_bit:default_n_bit_quantizers",
    ],
)

py_test(
    name = "default_n_bit_quantize_registry_test",
    srcs = [
        "default_n_bit_quantize_registry_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":default_n_bit_quantize_registry",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
    ],
)

py_library(
    name = "default_n_bit_transforms",
    srcs = [
        "default_n_bit_transforms.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_aware_activation",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_layer",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/experimental/default_n_bit:default_n_bit_quantize_configs",
        "//tensorflow_model_optimization/python/core/quantization/keras/experimental/default_n_bit:default_n_bit_quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:transforms",
    ],
)

py_strict_library(
    name = "default_n_bit_quantize_layout_transform",
    srcs = [
        "default_n_bit_quantize_layout_transform.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":default_n_bit_transforms",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_layout_transform",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:model_transformer",
    ],
)

py_strict_test(
    name = "default_n_bit_transforms_test",
    size = "large",
    srcs = [
        "default_n_bit_transforms_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":default_n_bit_quantize_configs",
        ":default_n_bit_transforms",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_aware_activation",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_layer",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:model_transformer",
        "//tensorflow_model_optimization/python/core/quantization/keras/layers:conv_batchnorm_test_utils",
    ],
)

py_strict_library(
    name = "default_n_bit_quantize_scheme",
    srcs = [
        "default_n_bit_quantize_scheme.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":default_n_bit_quantize_layout_transform",
        ":default_n_bit_quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_scheme",
    ],
)
