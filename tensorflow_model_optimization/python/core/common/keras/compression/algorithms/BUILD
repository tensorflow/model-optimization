load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library", "py_strict_test")
load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "pytype_strict_library")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

pytype_strict_library(
    name = "epr",
    srcs = ["epr.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        # tensorflow_compression dep1,
        "//tensorflow_model_optimization/python/core/common/keras/compression:algorithm",
    ],
)

py_strict_test(
    name = "epr_test",
    timeout = "long",
    srcs = ["epr_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "manual",  # TODO(b/284376360): Remove manual after resolve kokoro test deps issue.
        "requires-net:external",
    ],
    deps = [
        ":epr",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
    ],
)

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
    tags = ["requires-net:external"],
    deps = [
        ":same_training_and_inference",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
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
    tags = ["requires-net:external"],
    deps = [
        ":different_training_and_inference",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
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
    tags = ["requires-net:external"],
    deps = [
        ":bias_only",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
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
    tags = ["requires-net:external"],
    deps = [
        ":weight_clustering",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
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
    tags = ["requires-net:external"],
    deps = [
        ":periodical_update_and_scheduling",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras/testing:test_utils_mnist",
    ],
)
