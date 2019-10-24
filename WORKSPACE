workspace(name = "tensorflow_model_optimization")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
# TensorFlow models repository for slim preprocessing.
http_archive(
    name = "research",
    strip_prefix = "models-master/research",
    urls = ["https://github.com/tensorflow/models/archive/master.zip"],
)
http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz",
    sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
)

load("@rules_python//python:pip.bzl", "pip_repositories", "pip_import")
pip_import(
    name = "pip_dependencies",
    requirements = "@tensorflow_model_optimization//:requirements.bazel.txt",
)
load("@pip_dependencies//:requirements.bzl", "pip_install")
pip_repositories()
pip_install()
