load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library", "py_strict_test")
# Placeholder: load py_library
# Placeholder: load py_test

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_strict_library(
    name = "default_8bit",
    srcs = [
        "__init__.py",
    ],
    deps = [],
)

py_strict_library(
    name = "default_8bit_quantizers",
    srcs = [
        "default_8bit_quantizers.py",
    ],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
    ],
)

# TODO(b/185543626): Fix keras dependency
py_test(
    name = "default_8bit_quantizers_test",
    srcs = [
        "default_8bit_quantizers_test.py",
    ],
    deps = [
        ":default_8bit_quantizers",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_library(
    name = "default_8bit_quantize_configs",
    srcs = [
        "default_8bit_quantize_configs.py",
    ],
    deps = [
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_config",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
    ],
)

py_strict_library(
    name = "default_8bit_quantize_registry",
    srcs = [
        "default_8bit_quantize_registry.py",
    ],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_config",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_configs",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantizers",
    ],
)

# TODO(b/185543626): Fix keras dependency
py_test(
    name = "default_8bit_quantize_registry_test",
    srcs = [
        "default_8bit_quantize_registry_test.py",
    ],
    deps = [
        ":default_8bit_quantize_registry",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
    ],
)

# TODO(b/185543626): Fix keras dependency
py_library(
    name = "default_8bit_transforms",
    srcs = [
        "default_8bit_transforms.py",
    ],
    deps = [
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_aware_activation",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_layer",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_configs",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:transforms",
    ],
)

py_strict_test(
    name = "default_8bit_transforms_test",
    size = "large",
    srcs = [
        "default_8bit_transforms_test.py",
    ],
    deps = [
        ":default_8bit_quantize_configs",
        ":default_8bit_quantize_registry",
        ":default_8bit_transforms",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_aware_activation",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_layer",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:model_transformer",
        "//tensorflow_model_optimization/python/core/quantization/keras/layers:conv_batchnorm_test_utils",
        "//tensorflow_model_optimization/python/core/quantization/keras/layers:dense_batchnorm_test_utils",
    ],
)

py_strict_library(
    name = "default_8bit_quantize_layout_transform",
    srcs = [
        "default_8bit_quantize_layout_transform.py",
    ],
    deps = [
        ":default_8bit_transforms",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_layout_transform",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:model_transformer",
    ],
)

# TODO(b/185543626): Fix keras dependency
py_test(
    name = "quantize_numerical_test",
    size = "large",
    srcs = ["quantize_numerical_test.py"],
    deps = [
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
        "//tensorflow_model_optimization/python/core/quantization/keras:utils",
    ],
)

py_strict_library(
    name = "default_8bit_quantize_scheme",
    srcs = [
        "default_8bit_quantize_scheme.py",
    ],
    deps = [
        ":default_8bit_quantize_layout_transform",
        ":default_8bit_quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_scheme",
    ],
)
