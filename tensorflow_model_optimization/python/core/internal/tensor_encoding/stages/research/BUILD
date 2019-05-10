package(default_visibility = [
    "//tensorflow_model_optimization/python/core/internal/tensor_encoding:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "research",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":clipping",
        ":kashin",
    ],
)

py_library(
    name = "clipping",
    srcs = ["clipping.py"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:encoding_stage",
    ],
)

py_test(
    name = "clipping_test",
    size = "small",
    srcs = ["clipping_test.py"],
    deps = [
        ":clipping",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)

py_library(
    name = "kashin",
    srcs = ["kashin.py"],
    deps = [
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:encoding_stage",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:tf_utils",
    ],
)

py_test(
    name = "kashin_test",
    size = "medium",
    srcs = ["kashin_test.py"],
    deps = [
        ":kashin",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)
