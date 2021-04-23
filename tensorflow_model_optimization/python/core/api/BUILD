load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_strict_library(
    name = "api",
    srcs = glob(["**/__init__.py"]),
    srcs_version = "PY3",
    deps = [
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
        "//tensorflow_model_optimization/python/core/clustering/keras:clusterable_layer",
        "//tensorflow_model_optimization/python/core/clustering/keras:clustering_algorithm",
        "//tensorflow_model_optimization/python/core/clustering/keras:clustering_callbacks",
        "//tensorflow_model_optimization/python/core/clustering/keras/experimental:cluster",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_config",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_layout_transform",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_scheme",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_wrapper",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/collaborative_optimizations/cluster_preserve:cluster_utils",
        "//tensorflow_model_optimization/python/core/quantization/keras/collaborative_optimizations/cluster_preserve:default_8bit_cluster_preserve_quantize_scheme",
        "//tensorflow_model_optimization/python/core/quantization/keras/collaborative_optimizations/prune_preserve:default_8bit_prune_preserve_quantize_scheme",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_layout_transform",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_scheme",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_transforms",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:model_transformer",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:transforms",
        "//tensorflow_model_optimization/python/core/sparsity/keras:prunable_layer",
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_callbacks",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_schedule",
    ],
)
