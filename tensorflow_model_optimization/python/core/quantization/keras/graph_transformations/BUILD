load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library", "py_strict_test")

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_strict_library(
    name = "graph_transformations",
    srcs = [
        "__init__.py",
    ],
    deps = [],
)

py_strict_library(
    name = "transforms",
    srcs = [
        "transforms.py",
    ],
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
    visibility = ["//visibility:public"],
    deps = [
        ":transforms",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
    ],
)

py_strict_library(
    name = "model_transformer",
    srcs = [
        "model_transformer.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":transforms",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_test(
    name = "model_transformer_test",
    srcs = [
        "model_transformer_test.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":model_transformer",
        ":transforms",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras:utils",
    ],
)
