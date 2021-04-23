# buildifier: disable=same-origin-load
load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

# buildifier: disable=same-origin-load
load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_test")

package(default_visibility = [
    "//tensorflow_model_optimization/python/core/internal/tensor_encoding:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_strict_library(
    name = "encoders",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common_encoders",
    ],
)

py_strict_library(
    name = "common_encoders",
    srcs = ["common_encoders.py"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:core_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:gather_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:simple_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/stages:stages_impl",
    ],
)

py_strict_test(
    name = "common_encoders_test",
    size = "small",
    srcs = ["common_encoders_test.py"],
    python_version = "PY3",
    deps = [
        ":common_encoders",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:core_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:gather_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:simple_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:py_utils",
    ],
)
