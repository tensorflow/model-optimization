load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "pytype_strict_library", "pytype_strict_test")

licenses(["notice"])

pytype_strict_library(
    name = "algorithm",
    srcs = ["algorithm.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/common/keras/compression/internal:optimize",
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

pytype_strict_library(
    name = "schedules",
    srcs = ["schedules.py"],
    srcs_version = "PY3ONLY",
    deps = [
        # tensorflow dep1,
    ],
)

pytype_strict_test(
    name = "schedules_test",
    srcs = [
        "schedules_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":schedules",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
    ],
)
