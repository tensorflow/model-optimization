<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfmot.sparsity.keras" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfmot.sparsity.keras

Module containing sparsity code built on Keras abstractions.

Defined in
[`python/core/api/sparsity/keras/__init__.py`](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/api/sparsity/keras/__init__.py).

<!-- Placeholder for "Used in" -->

## Classes

[`class ConstantSparsity`](../../tfmot/sparsity/keras/ConstantSparsity.md):
Pruning schedule with constant sparsity(%) throughout training.

[`class PolynomialDecay`](../../tfmot/sparsity/keras/PolynomialDecay.md):
Pruning Schedule with a PolynomialDecay function.

[`class PrunableLayer`](../../tfmot/sparsity/keras/PrunableLayer.md): Abstract
Base Class for making your own keras layer prunable.

[`class PruningSchedule`](../../tfmot/sparsity/keras/PruningSchedule.md):
Specifies when to prune layer and the sparsity(%) at each training step.

[`class PruningSummaries`](../../tfmot/sparsity/keras/PruningSummaries.md): A
Keras callback for adding pruning summaries to tensorboard.

[`class UpdatePruningStep`](../../tfmot/sparsity/keras/UpdatePruningStep.md):
Keras callback which updates pruning wrappers with the optimizer step.

## Functions

[`prune_low_magnitude(...)`](../../tfmot/sparsity/keras/prune_low_magnitude.md):
Modify a keras layer or model to be pruned during training.

[`prune_scope(...)`](../../tfmot/sparsity/keras/prune_scope.md): Provides a
scope in which Pruned layers and models can be deserialized.

[`strip_pruning(...)`](../../tfmot/sparsity/keras/strip_pruning.md): Strip
pruning wrappers from the model.
