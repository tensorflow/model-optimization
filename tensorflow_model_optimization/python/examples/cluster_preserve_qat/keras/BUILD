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
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
        "//tensorflow_model_optimization/python/core/quantization/keras/collab_opts/cluster_preserve:cluster_utils",
        "//tensorflow_model_optimization/python/core/quantization/keras/collab_opts/cluster_preserve:default_8bit_cluster_preserve_quantize_scheme",
    ],
)
