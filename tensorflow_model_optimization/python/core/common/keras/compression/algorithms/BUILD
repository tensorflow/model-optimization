load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library", "py_strict_test")
load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "pytype_strict_library")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

pytype_strict_library(
    name = "same_training_and_inference",
    srcs = ["same_training_and_inference.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/common/keras/compression:algorithm",
    ],
)

py_strict_test(
    name = "same_training_and_inference_test",
    timeout = "long",
    srcs = ["same_training_and_inference_test.py"],
    python_version = "PY3",
    deps = [
        ":same_training_and_inference",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras/testing:test_utils_mnist",
    ],
)

pytype_strict_library(
    name = "different_training_and_inference",
    srcs = ["different_training_and_inference.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/common/keras/compression:algorithm",
    ],
)

py_strict_test(
    name = "different_training_and_inference_test",
    timeout = "long",
    srcs = ["different_training_and_inference_test.py"],
    python_version = "PY3",
    deps = [
        ":different_training_and_inference",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras/testing:test_utils_mnist",
    ],
)

pytype_strict_library(
    name = "bias_only",
    srcs = ["bias_only.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/common/keras/compression:algorithm",
    ],
)

py_strict_test(
    name = "bias_only_test",
    timeout = "long",
    srcs = ["bias_only_test.py"],
    python_version = "PY3",
    deps = [
        ":bias_only",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras/testing:test_utils_mnist",
    ],
)

py_strict_library(
    name = "weight_clustering",
    srcs = ["weight_clustering.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
        "//tensorflow_model_optimization/python/core/clustering/keras:clustering_centroids",
        "//tensorflow_model_optimization/python/core/clustering/keras:clustering_registry",
        "//tensorflow_model_optimization/python/core/common/keras/compression:algorithm",
    ],
)

py_strict_test(
    name = "weight_clustering_test",
    srcs = ["weight_clustering_test.py"],
    python_version = "PY3",
    deps = [
        ":weight_clustering",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
    ],
)

pytype_strict_library(
    name = "periodical_update_and_scheduling",
    srcs = ["periodical_update_and_scheduling.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/common/keras/compression:algorithm",
    ],
)

py_strict_test(
    name = "periodical_update_and_scheduling_test",
    timeout = "long",
    srcs = ["periodical_update_and_scheduling_test.py"],
    python_version = "PY3",
    deps = [
        ":periodical_update_and_scheduling",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras/testing:test_utils_mnist",
    ],
)
