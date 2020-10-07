package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
    "//third_party/tensorflow:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":compress",
    ],
)

py_library(
    name = "compress",
    srcs = ["compress.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":compress_wrapper",
        # attr dep1,
        # enum dep1,
        # tensorflow dep1,
        # python:util tensorflow dep2,
        # python/keras:models tensorflow dep2,
        # python/keras/engine tensorflow dep2,
    ],
)

py_test(
    name = "compress_test",
    srcs = [
        "compress_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":compress",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "compress_wrapper",
    srcs = [
        "compress_wrapper.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep1,
    ],
)

py_test(
    name = "compress_wrapper_test",
    srcs = [
        "compress_wrapper_test.py",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":compress_wrapper",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
    ],
)
