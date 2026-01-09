load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_test")
# Placeholder: load py_library

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "keras",
    srcs = ["__init__.py"],
    strict_deps = True,
    deps = [
        ":compat",
        ":utils",
    ],
)

# TODO(alanchiao): split into test_utils and utils.
py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    strict_deps = True,
    deps = [
        ":compat",
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "compat",
    srcs = ["compat.py"],
    strict_deps = True,
    deps = [
        # tensorflow dep1,
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        # tensorflow dep1,
        # python/framework:smart_cond tensorflow dep2,
        # python/ops:variables tensorflow dep2,
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    strict_deps = True,
    deps = [
        # python/eager:monitoring tensorflow dep2,
    ],
)

py_strict_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        ":compat",
        ":metrics",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # mock dep1,
        # tensorflow dep1,
        # python/eager:monitoring tensorflow dep2,
    ],
)
