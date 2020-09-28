licenses(["notice"])  # Apache 2.0

py_binary(
    name = "mnist_cnn",
    srcs = [
        "mnist_cnn.py",
    ],
    python_version = "PY3",
    deps = [
        # absl:app dep1,
        # numpy dep1,
        # tensorflow dep1,
        # python/keras tensorflow dep2,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
    ],
)
