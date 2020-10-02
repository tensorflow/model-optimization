package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "keras",
    srcs = ["__init__.py"],
    deps = [
        ":compat",
        ":utils",
    ],
)

# TODO(alanchiao): split into test_utils and utils.
py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "compat",
    srcs = ["compat.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        # python:control_flow_ops tensorflow dep2,
        # python:smart_cond tensorflow dep2,
        # python:variables tensorflow dep2,
    ],
)
