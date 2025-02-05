load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "pytype_strict_library")

package(default_visibility = [
    "//tensorflow_model_optimization/python/core/common/keras/compression:__subpackages__",
])

licenses(["notice"])

pytype_strict_library(
    name = "optimize",
    srcs = ["optimize.py"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)
