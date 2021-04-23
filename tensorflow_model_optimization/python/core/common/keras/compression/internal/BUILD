load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "pytype_library")

package(default_visibility = [
    "//tensorflow_model_optimization/python/core/common/keras/compression:__subpackages__",
])

licenses(["notice"])

pytype_library(
    name = "optimize",
    srcs = ["optimize.py"],
    srcs_version = "PY3",
    deps = [
        #TODO(b/185726660): remove when we stick to Python 3.7, which includes this by default.
        # dataclasses dep1,
        # tensorflow dep1,
    ],
)
