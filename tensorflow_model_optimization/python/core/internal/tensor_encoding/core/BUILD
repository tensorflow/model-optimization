load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_test")
# Placeholder: load py_library

package(default_visibility = [
    "//tensorflow_model_optimization/python/core/internal/tensor_encoding:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "core",
    srcs = ["__init__.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":core_encoder",
        ":encoding_stage",
        ":gather_encoder",
        ":simple_encoder",
    ],
)

py_library(
    name = "core_encoder",
    srcs = ["core_encoder.py"],
    strict_deps = True,
    deps = [
        ":encoding_stage",
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:py_utils",
    ],
)

py_strict_test(
    name = "core_encoder_test",
    size = "small",
    srcs = ["core_encoder_test.py"],
    deps = [
        ":core_encoder",
        ":encoding_stage",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)

py_library(
    name = "encoding_stage",
    srcs = ["encoding_stage.py"],
    strict_deps = True,
    deps = [
        # six dep1,
        # tensorflow dep1,
    ],
)

py_strict_test(
    name = "encoding_stage_test",
    size = "small",
    srcs = ["encoding_stage_test.py"],
    deps = [
        ":encoding_stage",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # mock dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)

py_library(
    name = "gather_encoder",
    srcs = ["gather_encoder.py"],
    strict_deps = True,
    deps = [
        ":core_encoder",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:py_utils",
    ],
)

py_strict_test(
    name = "gather_encoder_test",
    size = "small",
    srcs = ["gather_encoder_test.py"],
    deps = [
        ":core_encoder",
        ":encoding_stage",
        ":gather_encoder",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        # python/framework:test_lib tensorflow dep2,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)

py_library(
    name = "simple_encoder",
    srcs = ["simple_encoder.py"],
    strict_deps = True,
    deps = [
        ":core_encoder",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:py_utils",
    ],
)

py_strict_test(
    name = "simple_encoder_test",
    size = "small",
    srcs = ["simple_encoder_test.py"],
    deps = [
        ":core_encoder",
        ":simple_encoder",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        # python/framework:test_lib tensorflow dep2,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)
