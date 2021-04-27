# buildifier: disable=same-origin-load
load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

# buildifier: disable=same-origin-load
load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_test")
load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "pytype_strict_library")

licenses(["notice"])

pytype_strict_library(
    name = "algorithm",
    srcs = ["algorithm.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_model_optimization/python/core/common/keras/compression/internal:optimize",
        #TODO(b/185726660): remove when we stick to Python 3.7, which includes this by default.
        # dataclasses dep1,
        # tensorflow dep1,
    ],
)

py_strict_library(
    name = "schedules",
    srcs = ["schedules.py"],
    srcs_version = "PY3ONLY",
    deps = [
        # tensorflow dep1,
    ],
)

py_strict_test(
    name = "schedules_test",
    srcs = [
        "schedules_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":schedules",
        # tensorflow dep1,
    ],
)
