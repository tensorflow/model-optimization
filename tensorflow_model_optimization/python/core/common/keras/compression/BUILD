licenses(["notice"])

py_library(
    name = "algorithm",
    srcs = ["algorithm.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_model_optimization/python/core/common/keras/compression/internal:optimize",
        #TODO(tfmot): remove when we stick to Python 3.7, which includes this by default.
        # dataclasses dep1,
        # tensorflow dep1,
    ],
)
