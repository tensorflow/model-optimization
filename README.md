# TensorFlow Model Optimization Toolkit

The **TensorFlow Model Optimization Toolkit** is a suite of tools that users,
both novice and advanced, can use to optimize machine learning models for
deployment and execution.

For an overview of this project and individual tools, the optimization gains,
and our roadmap refer to
[tensorflow.org/model_optimization](https://www.tensorflow.org/model_optimization).
The website also provides various tutorials and API docs.

The toolkit provides stable Python APIs.

## Installation

### Stable Builds

To install the latest version, run the following:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --user --upgrade tensorflow-model-optimization
```

For release details, see our
[release notes](https://github.com/tensorflow/model-optimization/releases).

TensorFlow Model Optimization requires either Tensorflow 1.x for versions 1.14+
or the nightly build of [TensorFlow](https://www.tensorflow.org/install) (pip
package `tf-nightly`). Note that for the nightly build, you need to use
tf.compat.v1 since 2.x is the default now.

Since TensorFlow is *not* included as a dependency of the TensorFlow Model
Optimization package (in `setup.py`), you must explicitly install the TensorFlow
package (`tf-nightly` or `tf-nightly-gpu`). This allows us to maintain one
package instead of separate packages for CPU and GPU-enabled TensorFlow.

### Installing from Source

You can also install from source. This requires the
[Bazel](https://bazel.build/) build system.

```shell
# To install dependencies on Ubuntu:
# sudo apt-get install bazel git python-pip
# For other platforms, see Bazel docs above.
git clone https://github.com/tensorflow/model-optimization.git
cd tensorflow_model_optimization
bazel build --copt=-O3 --copt=-march=native :pip_pkg
PKGDIR=$(mktemp -d)
./bazel-bin/pip_pkg $PKGDIR
pip install --user --upgrade $PKGDIR/*.whl
```

## Community

As part of TensorFlow, we're committed to fostering an open and welcoming
environment.

*   [GitHub](https://github.com/tensorflow/model-optimization/issues): Report
    bugs or make feature requests.
*   [TensorFlow Blog](https://medium.com/tensorflow): Stay up to date on content
    from the TensorFlow team and best articles from the community.
