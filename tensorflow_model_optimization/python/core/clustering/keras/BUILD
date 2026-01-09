load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_test")
# Placeholder: load py_library
# Placeholder: load py_test

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
    ],
    strict_deps = True,
    deps = [
        ":cluster",  # buildcleaner: keep
        ":clustering_callbacks",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core/clustering/keras/experimental",  # buildcleaner: keep
    ],
)

py_library(
    name = "cluster",
    srcs = ["cluster.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_config",
        ":cluster_wrapper",
        ":clustering_centroids",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_library(
    name = "cluster_config",
    srcs = ["cluster_config.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
)

py_library(
    name = "clustering_registry",
    srcs = ["clustering_registry.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":clusterable_layer",
        ":clustering_algorithm",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_library(
    name = "clusterable_layer",
    srcs = ["clusterable_layer.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
    ],
)

py_library(
    name = "clustering_centroids",
    srcs = ["clustering_centroids.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_config",
        # six dep1,
        # tensorflow dep1,
        # python/ops:clustering_ops tensorflow dep2,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_library(
    name = "cluster_wrapper",
    srcs = ["cluster_wrapper.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_config",
        ":clusterable_layer",
        ":clustering_centroids",
        ":clustering_registry",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_library(
    name = "clustering_algorithm",
    srcs = ["clustering_algorithm.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
    ],
)

py_library(
    name = "clustering_callbacks",
    srcs = ["clustering_callbacks.py"],
    strict_deps = True,
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
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_config",
        ":cluster_wrapper",
        ":clusterable_layer",
        ":clustering_registry",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras/experimental:cluster",
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_test(
    name = "clustering_centroids_test",
    size = "medium",
    srcs = ["clustering_centroids_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_config",
        ":clustering_centroids",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_test(
    name = "cluster_wrapper_test",
    size = "medium",
    srcs = ["cluster_wrapper_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_config",
        ":cluster_wrapper",
        ":clusterable_layer",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_test(
    name = "clustering_registry_test",
    size = "medium",
    srcs = ["clustering_registry_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_config",
        ":clusterable_layer",
        ":clustering_registry",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_test(
    name = "cluster_integration_test",
    size = "medium",
    srcs = ["cluster_integration_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_config",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras/experimental:cluster",
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_test(
    name = "cluster_distributed_test",
    srcs = ["cluster_distributed_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_config",
        ":cluster_wrapper",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras/experimental:cluster",
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras:test_utils",
    ],
)

py_strict_test(
    name = "mnist_clusterable_layer_test",
    srcs = ["mnist_clusterable_layer_test.py"],
    tags = ["requires-net:external"],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_config",
        ":clusterable_layer",
        ":clustering_algorithm",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_test(
    name = "mnist_clustering_test",
    srcs = ["mnist_clustering_test.py"],
    tags = ["requires-net:external"],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_config",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras/experimental:cluster",
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)
