load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library", "py_strict_test")

package(default_visibility = [
    "//tensorflow_model_optimization/python/core/internal/tensor_encoding:__subpackages__",
])

licenses(["notice"])

py_strict_library(
    name = "testing",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":test_utils",
    ],
)

py_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
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
    python_version = "PY3",
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
