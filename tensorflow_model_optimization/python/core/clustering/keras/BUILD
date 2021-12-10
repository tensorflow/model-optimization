load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library", "py_strict_test")

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_strict_library(
    name = "keras",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":cluster",  # buildcleaner: keep
        ":clustering_callbacks",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core/clustering/keras/experimental",  # buildcleaner: keep
    ],
)

py_strict_library(
    name = "cluster",
    srcs = ["cluster.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_config",
        ":cluster_wrapper",
        ":clustering_centroids",
        # tensorflow dep1,
    ],
)

py_strict_library(
    name = "cluster_config",
    srcs = ["cluster_config.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_strict_library(
    name = "clustering_registry",
    srcs = ["clustering_registry.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":clusterable_layer",
        ":clustering_algorithm",
        # tensorflow dep1,
    ],
)

py_strict_library(
    name = "clusterable_layer",
    srcs = ["clusterable_layer.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
    ],
)

py_strict_library(
    name = "clustering_centroids",
    srcs = ["clustering_centroids.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_config",
        # six dep1,
        # tensorflow dep1,
        # python:clustering_ops tensorflow dep2,
    ],
)

py_strict_library(
    name = "cluster_wrapper",
    srcs = ["cluster_wrapper.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_config",
        ":clusterable_layer",
        ":clustering_centroids",
        ":clustering_registry",
        # tensorflow dep1,
    ],
)

py_strict_library(
    name = "clustering_algorithm",
    srcs = ["clustering_algorithm.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
    ],
)

py_strict_library(
    name = "clustering_callbacks",
    srcs = ["clustering_callbacks.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_test(
    name = "cluster_test",
    size = "medium",
    srcs = ["cluster_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_config",
        ":cluster_wrapper",
        ":clusterable_layer",
        ":clustering_registry",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras/experimental:cluster",
    ],
)

py_strict_test(
    name = "clustering_centroids_test",
    size = "medium",
    srcs = ["clustering_centroids_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_config",
        ":clustering_centroids",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_strict_test(
    name = "cluster_wrapper_test",
    size = "medium",
    srcs = ["cluster_wrapper_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_config",
        ":cluster_wrapper",
        ":clusterable_layer",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
    ],
)

py_strict_test(
    name = "clustering_registry_test",
    size = "medium",
    srcs = ["clustering_registry_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_config",
        ":clusterable_layer",
        ":clustering_registry",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
    ],
)

py_test(
    name = "cluster_integration_test",
    size = "medium",
    srcs = ["cluster_integration_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_config",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras/experimental:cluster",
    ],
)

py_strict_test(
    name = "cluster_distributed_test",
    srcs = ["cluster_distributed_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_config",
        ":cluster_wrapper",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras/experimental:cluster",
        "//tensorflow_model_optimization/python/core/keras:test_utils",
    ],
)

py_strict_test(
    name = "mnist_clusterable_layer_test",
    srcs = ["mnist_clusterable_layer_test.py"],
    python_version = "PY3",
    tags = ["requires-net:external"],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_config",
        ":clusterable_layer",
        ":clustering_algorithm",
        # tensorflow dep1,
    ],
)

py_strict_test(
    name = "mnist_clustering_test",
    srcs = ["mnist_clustering_test.py"],
    python_version = "PY3",
    tags = ["requires-net:external"],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_config",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras/experimental:cluster",
    ],
)
