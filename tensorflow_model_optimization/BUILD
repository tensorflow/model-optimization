# Description:
#
#  TensorFlow Optimization is a repository for the training-time
#  portion of the Tensorflow Model Optimization Toolkit,
#  used to optimize machine learning models for deployment and execution.
#
#  https://github.com/tensorflow/model-optimization
load("//tensorflow_model_optimization:tensorflow_model_optimization.bzl", "py_strict_binary", "py_strict_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

py_strict_library(
    name = "tensorflow_model_optimization",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_model_optimization/python",  # buildcleaner: keep
        "//tensorflow_model_optimization/python/core:version",
        "//tensorflow_model_optimization/python/core/api",
    ],
)

py_strict_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    python_version = "PY3",
    deps = [
        # absl:app dep1,
        # absl/flags dep1,
        # google/protobuf:use_fast_cpp_protos dep1,  # Automatically added
        # tensorflow_docs/api_generator:generate_lib dep1,
        "//third_party/tensorflow_model_optimization",
    ],
)
