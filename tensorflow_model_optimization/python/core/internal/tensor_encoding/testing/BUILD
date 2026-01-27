load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_test")
# Placeholder: load py_library

package(default_visibility = [
    "//tensorflow_model_optimization/python/core/internal/tensor_encoding:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "testing",
    srcs = ["__init__.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":test_utils",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    strict_deps = True,
    deps = [
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:encoding_stage",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:py_utils",
    ],
)

py_strict_test(
    name = "test_utils_test",
    size = "medium",
    srcs = ["test_utils_test.py"],
    deps = [
        ":test_utils",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # mock dep1,
        # numpy dep1,
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:encoding_stage",
    ],
)
