package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "cluster_preserve",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
    ],
)

py_library(
    name = "cluster_utils",
    srcs = [
        "cluster_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:clustering_registry",
    ],
)

py_library(
    name = "cluster_preserve_quantize_registry",
    srcs = [
        "cluster_preserve_quantize_registry.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":cluster_utils",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantizers",
    ],
)

py_test(
    name = "cluster_preserve_quantize_registry_test",
    srcs = [
        "cluster_preserve_quantize_registry_test.py",
    ],
    python_version = "PY3",
    visibility = ["//visibility:private"],
    deps = [
        ":cluster_preserve_quantize_registry",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_config",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
    ],
)

py_library(
    name = "default_8bit_cluster_preserve_quantize_scheme",
    srcs = [
        "default_8bit_cluster_preserve_quantize_scheme.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":cluster_preserve_quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_scheme",
    ],
)
