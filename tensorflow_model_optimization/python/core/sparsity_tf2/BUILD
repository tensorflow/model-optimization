package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "sparsity_tf2",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":lthpruner",
        ":prunable_model",
        ":pruner",
        ":pruning_config",
        ":pruning_optimizer",
    ],
)

py_library(
    name = "prunable_model",
    srcs = ["prunable_model.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruning_optimizer",
        # tensorflow dep1,
    ],
)

py_library(
    name = "pruning_config",
    srcs = ["pruning_config.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_schedule",
        "//tensorflow_model_optimization/python/core/sparsity/keras:prunable_layer",
        "//tensorflow_model_optimization/python/core/sparsity/keras:prune_registry",
        ":pruner",
        # tensorflow dep1,
    ],
)

py_library(
    name = "pruning_optimizer",
    srcs = ["pruning_optimizer.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # tensorflow dep1,
    ],
)

py_library(
    name = "pruner",
    srcs = ["pruner.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_utils",
        # tensorflow dep1,
        # python:summary tensorflow dep2,
    ],
)

py_library(
    name = "lthpruner",
    srcs = ["lthpruner.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_utils",
        # tensorflow dep1,
        # python:summary tensorflow dep2,
    ],
)

py_library(
    name = "riglpruner",
    srcs = ["riglpruner.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_utils",
        "//tensorflow_model_optimization/python/core/sparsity/keras:sparse_utils",
        # tensorflow dep1,
        # python:summary tensorflow dep2,
    ],
)

py_test(
    name = "pruner_test",
    size = "medium",
    srcs = ["pruner_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pruner",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_schedule",
        # numpy dep1,
        # python/keras tensorflow dep2,
    ],
)

py_test(
    name = "lthpruner_test",
    size = "medium",
    srcs = ["lthpruner_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":lthpruner",
        ":pruner",
        "//tensorflow_model_optimization/python/core/sparsity/keras:pruning_schedule",
        # numpy dep1,
        # python/keras tensorflow dep2,
    ],
)

py_test(
    name = "riglpruner_test",
    size = "medium",
    srcs = ["riglpruner_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":riglpruner",
        ":pruner",
        "//tensorflow_model_optimization/python/core/sparsity/keras:schedule",
        # numpy dep1,
        # python/keras tensorflow dep2,
    ],
)