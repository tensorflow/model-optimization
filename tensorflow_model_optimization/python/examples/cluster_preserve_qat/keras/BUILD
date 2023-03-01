load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_binary")

licenses(["notice"])

py_strict_binary(
    name = "mnist_cnn",
    srcs = [
        "mnist_cnn.py",
    ],
    python_version = "PY3",
    deps = [
        # absl:app dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
        "//tensorflow_model_optimization/python/core/quantization/keras/collab_opts/cluster_preserve:cluster_utils",
        "//tensorflow_model_optimization/python/core/quantization/keras/collab_opts/cluster_preserve:default_8bit_cluster_preserve_quantize_scheme",
    ],
)
