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


# TensorFlow Model Optimization 0.7.1

TFMOT 0.7.1 fixes a bug in tensor_encoding in 9e4c106267a4a7f61e0d90b0848db15fd063b80e.

# TensorFlow Model Optimization 0.7.0

TFMOT 0.7.0 adds updates for Quantization Aware Training (QAT)
and Pruning API. Adds support for structured (MxN) pruning.
QAT now also has support for layers with swish activations and ability
to disable per-axis quantization in the default8_bit scheme.
Adds support for combining pruning, QAT and weight clustering.

Keras Quantization API:
Tested against TensorFlow 2.6.0, 2.5.1 and nightly with Python 3.
* Added QuantizeWrapperV2 class which preserves order of weights is the default for quantize_apply.
* Added a flag to disable per-axis quantizers in default8_bit scheme.
* Added swish as supported activation.

Keras pruning API:
Tested against TensorFlow 2.6.0, 2.5.1 and nightly with Python 3.
* Added structural pruning with MxN sparsity.

Keras clustering API:
* Added support for RNNSimple, LSTM, GRU, StackedRNNCells, PeepholeLSTMCell, and Bidirectional layers.
* Updated and fixed sparsity-preserving clustering.
* Added an experimental quantization schemes for Quantization Aware Training for collaborative model.optimization:
    - Pruning-Clustering-preserving QAT: pruned and clustered model can be QAT trained with preserved sparsity and the number of clusters.
* Updated Clustering initialization default to KMEANS_PLUS_PLUS.

# TensorFlow Model Optimization 0.6.0

TFMOT 0.6.0 adds some additional features for Quantization Aware Training (QAT)
and Pruning API. Adds support for overriding and subclassing default quantization
schemes. Adds input quantizer for annotated quantized layers without annotated
input layers. QAT now also has support for Conv2DTranspose and tanh layers.
For Pruning API, added pruning policy for pruning registries targeting specific
hardware.

Keras quantization API:
Tested against TensorFlow 2.4.2, 2.5.0 and nightly with Python 3.

Keras pruning API:
Tested against TensorFlow 2.4.2, 2.5.0 and nightly with Python 3.

Keras clustering API:
* Added *ClusteringSummaries* to create additional output for the clustering
progress for TensorBoard.
* Added ClusterableLayer API to support clustering of a keras custom layer.
In addition, now clustering can be done for bias of the layer.
* Introduced two new experimental quantization schemes for Quantization Aware Training
for collaborative model optimization:
    - Prune Preserve QAT: pruned model can be QAT trained with preserved sparsity;
    - Cluster Preserve QAT: clustered model can be QAT trained with preserved clustering;
* Added a new feature to clustering: average gradient aggregation, which can
improve performance for some models.
* Updated clustering results in the documentation.
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
