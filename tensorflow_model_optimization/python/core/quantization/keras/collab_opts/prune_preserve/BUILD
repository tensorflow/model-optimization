# Placeholder: load py_library
# Placeholder: load py_test

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "prune_preserve",
    srcs = [
        "__init__.py",
    ],
    strict_deps = True,
    deps = [
        ":default_8bit_prune_preserve_quantize_scheme",  # buildcleaner: keep
    ],
)

py_library(
    name = "prune_preserve_quantize_registry",
    srcs = [
        "prune_preserve_quantize_registry.py",
    ],
    strict_deps = True,
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
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
    visibility = ["//visibility:private"],
    deps = [
        ":prune_preserve_quantize_registry",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune_registry",
    ],
)

py_library(
    name = "default_8bit_prune_preserve_quantize_scheme",
    srcs = [
        "default_8bit_prune_preserve_quantize_scheme.py",
    ],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":prune_preserve_quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_scheme",
    ],
)
