load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library", "py_strict_test")

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_strict_library(
    name = "graph_transformations",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [],
)

py_strict_library(
    name = "transforms",
    srcs = [
        "transforms.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
    ],
)

py_strict_test(
    name = "transforms_test",
    srcs = [
        "transforms_test.py",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":transforms",
        # tensorflow dep1,
    ],
)

py_strict_library(
    name = "model_transformer",
    srcs = [
        "model_transformer.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":transforms",
        # tensorflow dep1,
    ],
)

py_strict_test(
    name = "model_transformer_test",
    srcs = [
        "model_transformer_test.py",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":model_transformer",
        ":transforms",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
    ],
)
