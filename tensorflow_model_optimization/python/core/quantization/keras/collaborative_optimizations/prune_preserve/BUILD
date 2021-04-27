load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_strict_library(
    name = "prune_preserve",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":default_8bit_prune_preserve_quantize_scheme",  # buildcleaner: keep
    ],
)

py_strict_library(
    name = "prune_preserve_quantize_registry",
    srcs = [
        "prune_preserve_quantize_registry.py",
    ],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quant_ops",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantizers",
    ],
)

py_test(
    name = "prune_preserve_quantize_registry_test",
    srcs = [
        "prune_preserve_quantize_registry_test.py",
    ],
    python_version = "PY3",
    visibility = ["//visibility:private"],
    deps = [
        ":prune_preserve_quantize_registry",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune_registry",
    ],
)

py_strict_library(
    name = "default_8bit_prune_preserve_quantize_scheme",
    srcs = [
        "default_8bit_prune_preserve_quantize_scheme.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":prune_preserve_quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_scheme",
    ],
)
