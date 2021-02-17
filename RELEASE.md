<!--
See https://github.com/tensorflow/model-optimization/releases/ for previous
examples of release notes. This project follows https://semver.org/.

"Tested against" references the versions of TensorFlow that TFMOT unit tests
will be run against prior to release. For 2.X, only the earliest (2.0.0) and latest (nightly)
TF releases are tested against, under the assumption that everything in between
works sufficiently well enough.
-->


# Release Template

Keras clustering API:

* Major Features
* Bug Fixes and Other Changes
* Tested against TensorFlow 1.14.0, 2.0.0, and nightly, and Python 3.

Keras quantization API:

* Major Features and Improvements
* Bug Fixes and Other Changes
* Tested against TensorFlow nightly, and Python 3.

Keras pruning API:

* Major Features and Improvements
* Bug Fixes and Other Changes
* Tested against TensorFlow 1.14.0, 2.0.0, and nightly, and Python 3.


# TensorFlow Model Optimization next release TBD

Keras clustering API:

* Added *ClusteringSummaries* to create additional output for the clustering
progress for TensorBoard.
* Tested against TensorFlow 1.14.0, 2.0.0, and nightly, and Python 3.

# TensorFlow Model Optimization 0.5.0

TFMOT 0.5.0 adds some additional features for Quantization Aware Training. QAT
now supports Keras layers SeparableConv2D and SeparableConv1D. It also provides
a new Quantizer AllValuesQuantizer which allows for more flexibility with range
selection.

Keras clustering API:
Tested against TensorFlow 1.14.0 and 2.3.0 with Python 3.

Keras quantization API:
Tested against TensorFlow 2.3.0 with Python 3.

Keras pruning API:
Tested against TensorFlow 1.14.0 and 2.3.0 with Python 3.


# TensorFlow Model Optimization 0.4.1

TFMOT 0.4.1 fixes a bug which makes 0.4.0 quantization code fail when run
against tf-nightly since July 31, 2020. The code now works against different
versions on TF, and is not broken by changes to smart_cond in core TF.

Keras clustering API:

Tested against TensorFlow 1.14.0, 2.0.0, and nightly, and Python 3.
Keras quantization API:

Tested against TensorFlow nightly, and Python 3.
Keras pruning API:

Tested against TensorFlow 1.14.0, 2.0.0, and nightly, and Python 3.
Pruning now doesn't remove the last remaining connection. So extreme sparsities like 0.999.. would remove all connections but one.


# TensorFlow Model Optimization 0.4.0

TFMOT 0.4.0 is the last release to support Python 2. Python 2 support officially
ended on January 1, 2020 and TF 2.1.0 was the last release to support Python 2.

Keras clustering API:

New API for weight clustering
Major Features
Support for clustering convolutional (except DepthwiseConv), Dense and other commonly used standard Keras layers
Support for different initialization methods for the cluster centroids: density-based, linear, random
Fine-tuning of cluster centroids during training
Tested against TensorFlow 1.14.0, 2.0.0, and nightly, and Python 3.
Keras quantization API:

Bug Fixes and Other Changes
Fixed Sequential model support for BatchNorm layers that follow Conv/DepthwiseConv (issue).
Improved error message for not using quantize_scope with custom Keras layers and objects.
Tested against TensorFlow nightly, and Python 2/3.
Keras pruning API:

Tested against TensorFlow 1.14.0, 2.0.0, and nightly, and Python2/3.
