# Placeholder: load py_library

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "layers",
    srcs = [
        "__init__.py",
    ],
    strict_deps = True,
    deps = [],
)

py_library(
    name = "conv_batchnorm_test_utils",
    srcs = ["conv_batchnorm_test_utils.py"],
    strict_deps = True,
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_library(
    name = "dense_batchnorm_test_utils",
    srcs = ["dense_batchnorm_test_utils.py"],
    strict_deps = True,
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)
