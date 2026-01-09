# Placeholder: load py_library

package(default_visibility = ["//tensorflow_model_optimization:__subpackages__"])

licenses(["notice"])

py_library(
    name = "collab_opts",
    srcs = ["__init__.py"],
    strict_deps = True,
    deps = [
        "//tensorflow_model_optimization/python/core/quantization/keras/collab_opts/cluster_preserve",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core/quantization/keras/collab_opts/prune_preserve",  # buildcleaner: keep
    ],
)
