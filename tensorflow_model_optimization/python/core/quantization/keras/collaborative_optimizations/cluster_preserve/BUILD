load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_strict_library(
    name = "cluster_preserve",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":default_8bit_cluster_preserve_quantize_scheme",  # buildcleaner: keep
    ],
)

py_strict_library(
    name = "cluster_utils",
    srcs = [
        "cluster_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
    ],
)

py_strict_library(
    name = "cluster_preserve_quantize_registry",
    srcs = [
        "cluster_preserve_quantize_registry.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":cluster_utils",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:clustering_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras:quant_ops",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
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
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:clustering_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_config",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
    ],
)

py_test(
    name = "mnist_prune_cluster_preserve_qat_test",
    srcs = [
        "mnist_prune_cluster_preserve_qat_test.py",
    ],
    python_version = "PY3",
    visibility = ["//visibility:private"],
    deps = [
        # tensorflow dep1,
        ":default_8bit_cluster_preserve_quantize_scheme",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_callbacks",
        "//tensorflow_model_optimization/python/core/clustering/keras/experimental:cluster",
    ]
)

py_strict_library(
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

py_test(
    name = "cluster_preserve_integration_test",
    srcs = [
        "cluster_preserve_integration_test.py",
    ],
    python_version = "PY3",
    visibility = ["//visibility:private"],
    deps = [
        ":default_8bit_cluster_preserve_quantize_scheme",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:clustering_registry",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras/experimental:cluster",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_config",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize"
    ],
)
