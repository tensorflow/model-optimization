load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library", "py_strict_test")

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_strict_library(
    name = "layers",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [],
)

py_strict_library(
    name = "conv_batchnorm",
    srcs = ["conv_batchnorm.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        # python:array_ops tensorflow dep2,
        # python:math_ops tensorflow dep2,
        # python:nn tensorflow dep2,
        # python:nn_ops tensorflow dep2,
        # python/framework:for_generated_wrappers tensorflow dep2,
        # python/keras:activations tensorflow dep2,
        # python/keras:backend tensorflow dep2,
        # python/keras/initializers tensorflow dep2,
        # python/keras/layers tensorflow dep2,
        # python/keras/layers:convolutional tensorflow dep2,
        # python/keras/layers/normalization:batch_normalization_v1 tensorflow dep2,
        # python/keras/utils:engine_utils tensorflow dep2,
        "//tensorflow_model_optimization/python/core/keras:utils",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantizers",
    ],
)

py_strict_test(
    name = "conv_batchnorm_test",
    srcs = ["conv_batchnorm_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conv_batchnorm",
        ":conv_batchnorm_test_utils",
        # numpy dep1,
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
        "//tensorflow_model_optimization/python/core/quantization/keras:utils",
    ],
)

py_strict_library(
    name = "conv_batchnorm_test_utils",
    srcs = ["conv_batchnorm_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":conv_batchnorm",
        # tensorflow dep1,
    ],
)
