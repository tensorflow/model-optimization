load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_binary")

licenses(["notice"])

py_strict_binary(
    name = "mnist_cnn",
    srcs = [
        "mnist_cnn.py",
    ],
    deps = [
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
    ],
)

py_strict_binary(
    name = "mnist_cnn_cont_quant",
    srcs = [
        "mnist_cnn_cont_quant.py",
    ],
    deps = [
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
    ],
)
