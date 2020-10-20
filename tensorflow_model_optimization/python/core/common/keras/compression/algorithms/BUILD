package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "same_training_and_inference",
    srcs = ["same_training_and_inference.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/common/keras/compression:algorithm",
    ],
)

py_test(
    name = "same_training_and_inference_test",
    srcs = ["same_training_and_inference_test.py"],
    python_version = "PY3",
    deps = [
        ":same_training_and_inference",
        # numpy dep1,
        # tensorflow dep1,
    ],
)
