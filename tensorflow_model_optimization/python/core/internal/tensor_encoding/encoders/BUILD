load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_test")
# Placeholder: load py_library

package(default_visibility = [
    "//tensorflow_model_optimization/python/core/internal/tensor_encoding:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "encoders",
    srcs = ["__init__.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":common_encoders",
    ],
)

py_library(
    name = "common_encoders",
    srcs = ["common_encoders.py"],
    strict_deps = True,
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
    deps = [
        ":common_encoders",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:core_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:gather_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:simple_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:py_utils",
    ],
)
