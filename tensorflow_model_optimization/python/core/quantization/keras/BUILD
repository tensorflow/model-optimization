load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_library", "py_strict_test")

package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
    "//third_party/tensorflow:__subpackages__",
])

licenses(["notice"])

py_strict_library(
    name = "keras",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":quantize",  # build-cleaner: keep
        # APIs are not exposed, but still needed for internal imports.
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core/quantization/keras/layers",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core/quantization/keras/experimental/default_n_bit",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core/quantization/keras/collab_opts",  # buildcleaner: keep
    ],
)

py_strict_library(
    name = "quant_ops",
    srcs = ["quant_ops.py"],
    srcs_version = "PY3",
    deps = [
        # absl/logging dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_test(
    name = "quant_ops_test",
    size = "small",
    srcs = ["quant_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":quant_ops",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_library(
    name = "quantizers",
    srcs = [
        "quantizers.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":quant_ops",
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_test(
    name = "quantizers_test",
    srcs = [
        "quantizers_test.py",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantizers",
        ":utils",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_library(
    name = "quantize_config",
    srcs = [
        "quantize_config.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantizers",
        ":utils",
        # six dep1,
        # tensorflow dep1,
    ],
)

py_strict_library(
    name = "quantize_registry",
    srcs = [
        "quantize_registry.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
    ],
)

py_strict_library(
    name = "quantize_layout_transform",
    srcs = [
        "quantize_layout_transform.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
    ],
)

py_strict_library(
    name = "quantize_annotate",
    srcs = [
        "quantize_annotate.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_test(
    name = "quantize_annotate_test",
    srcs = [
        "quantize_annotate_test.py",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_annotate",
        ":quantize_config",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_library(
    name = "quantize_aware_activation",
    srcs = [
        "quantize_aware_activation.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras:utils",
    ],
)

py_strict_test(
    name = "quantize_aware_activation_test",
    srcs = [
        "quantize_aware_activation_test.py",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_aware_activation",
        ":quantizers",
        ":utils",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_library(
    name = "quantize_layer",
    srcs = [
        "quantize_layer.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantizers",
        ":utils",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras:utils",
    ],
)

py_strict_test(
    name = "quantize_layer_test",
    srcs = [
        "quantize_layer_test.py",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_layer",
        ":quantizers",
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_library(
    name = "quantize_wrapper",
    srcs = [
        "quantize_wrapper.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_aware_activation",
        ":utils",
        # tensorflow dep1,
        # python/util:tf_inspect tensorflow dep2,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras:metrics",
        "//tensorflow_model_optimization/python/core/keras:utils",
    ],
)

py_strict_test(
    name = "quantize_wrapper_test",
    srcs = [
        "quantize_wrapper_test.py",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_aware_activation",
        ":quantize_wrapper",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
    ],
)

py_strict_library(
    name = "quantize",
    srcs = [
        "quantize.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_annotate",
        ":quantize_aware_activation",
        ":quantize_config",
        ":quantize_layer",
        ":quantize_wrapper",
        ":quantizers",
        ":utils",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras:metrics",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_scheme",
        "//tensorflow_model_optimization/python/core/quantization/keras/experimental/default_n_bit:default_n_bit_quantize_registry",
    ],
)

py_strict_test(
    name = "quantize_test",
    srcs = ["quantize_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize",
        ":quantize_annotate",
        ":quantize_config",
        ":quantize_layer",
        ":quantize_wrapper",
        ":quantizers",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras:test_utils",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
    ],
)

py_strict_test(
    name = "quantize_integration_test",
    srcs = ["quantize_integration_test.py"],
    python_version = "PY3",
    deps = [
        ":quantize",
        ":quantize_config",
        ":quantizers",
        ":utils",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras:test_utils",
    ],
)

py_strict_test(
    name = "quantize_models_test",
    size = "enormous",
    srcs = ["quantize_models_test.py"],
    flaky = True,
    python_version = "PY3",
    deps = [
        ":quantize",
        ":utils",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_test(
    name = "quantize_functional_test",
    size = "large",
    srcs = ["quantize_functional_test.py"],
    python_version = "PY3",
    deps = [
        ":quantize",
        ":utils",
        # absl/testing:parameterized dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras/testing:test_utils_mnist",
    ],
)

py_strict_library(
    name = "utils",
    srcs = [
        "utils.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_strict_library(
    name = "quantize_scheme",
    srcs = [
        "quantize_scheme.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
    ],
)
