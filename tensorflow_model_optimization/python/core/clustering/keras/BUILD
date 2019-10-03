package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cluster",
    ],
)

py_library(
    name = "cluster",
    srcs = ["cluster.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":clustering_centroids",
        ":clustering_registry",
        ":cluster_wrapper"
    ],
)

py_library(
    name = "clustering_registry",
    srcs = ["clustering_registry.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":clusterable_layer",
    ],
)

py_library(
    name = "clusterable_layer",
    srcs = ["clusterable_layer.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

py_library(
    name = "clustering_centroids",
    srcs = ["clustering_centroids.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

py_library(
    name = "cluster_wrapper",
    srcs = ["cluster_wrapper.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

py_test(
    name = "cluster_test",
    size = "medium",
    srcs = ["cluster_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
    ],
)

py_test(
    name = "clustering_centroids_test",
    size = "medium",
    srcs = ["clustering_centroids_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":clustering_centroids",
    ],
)

py_test(
    name = "cluster_wrapper_test",
    size = "medium",
    srcs = ["cluster_wrapper_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_wrapper",
        ":clusterable_layer",
        ":clustering_registry",
        ":clustering_centroids",
        ":cluster",
    ],
)

py_test(
    name = "clustering_registry_test",
    size = "medium",
    srcs = ["clustering_registry_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_wrapper",
        ":clusterable_layer",
        ":clustering_registry"
    ],
)