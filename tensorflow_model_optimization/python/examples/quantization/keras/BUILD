load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_binary")

licenses(["notice"])  # Apache 2.0

py_strict_binary(
    name = "mnist_cnn",
    srcs = [
        "mnist_cnn.py",
    ],
    python_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
    ],
)

py_strict_binary(
    name = "mnist_cnn_cont_quant",
    srcs = [
        "mnist_cnn_cont_quant.py",
    ],
    python_version = "PY3",
    deps = [
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
    ],
)
