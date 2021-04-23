# buildifier: disable=same-origin-load
load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

# buildifier: disable=same-origin-load
load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_test")

package(default_visibility = [
    "//tensorflow_model_optimization/python/core/internal/tensor_encoding:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_strict_library(
    name = "research",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":clipping",
        ":kashin",
        ":misc",
        ":quantization",
    ],
)

py_strict_library(
    name = "clipping",
    srcs = ["clipping.py"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:encoding_stage",
    ],
)

py_strict_test(
    name = "clipping_test",
    size = "small",
    srcs = ["clipping_test.py"],
    python_version = "PY3",
    deps = [
        ":clipping",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)

py_strict_library(
    name = "kashin",
    srcs = ["kashin.py"],
    deps = [
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:encoding_stage",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:tf_utils",
    ],
)

py_strict_test(
    name = "kashin_test",
    size = "medium",
    srcs = ["kashin_test.py"],
    python_version = "PY3",
    deps = [
        ":kashin",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)

py_strict_library(
    name = "quantization",
    srcs = ["quantization.py"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:encoding_stage",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:tf_utils",
    ],
)

py_strict_test(
    name = "quantization_test",
    size = "small",
    srcs = ["quantization_test.py"],
    python_version = "PY3",
    deps = [
        ":quantization",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)

py_strict_library(
    name = "misc",
    srcs = ["misc.py"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:encoding_stage",
    ],
)

py_strict_test(
    name = "misc_test",
    size = "small",
    srcs = ["misc_test.py"],
    python_version = "PY3",
    deps = [
        ":misc",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)
