load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_test")
# Placeholder: load py_library

package(default_visibility = [
    "//tensorflow_model_optimization/python/core/internal/tensor_encoding:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "stages",
    srcs = ["__init__.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":stages_impl",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/stages/research",
    ],
)

py_library(
    name = "stages_impl",
    srcs = ["stages_impl.py"],
    strict_deps = True,
    deps = [
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:encoding_stage",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:tf_utils",
    ],
)

py_strict_test(
    name = "stages_impl_test",
    size = "medium",
    srcs = ["stages_impl_test.py"],
    deps = [
        ":stages_impl",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)
