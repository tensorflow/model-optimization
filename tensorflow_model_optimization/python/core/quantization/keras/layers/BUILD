load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library")

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_strict_library(
    name = "layers",
    srcs = [
        "__init__.py",
    ],
    deps = [],
)

py_strict_library(
    name = "conv_batchnorm_test_utils",
    srcs = ["conv_batchnorm_test_utils.py"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_library(
    name = "dense_batchnorm_test_utils",
    srcs = ["dense_batchnorm_test_utils.py"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)
