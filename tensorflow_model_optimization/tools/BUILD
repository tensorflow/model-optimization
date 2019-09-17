package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow_docs/api_generator:generate_lib dep1,
        "//third_party/tensorflow_model_optimization",
    ],
)
