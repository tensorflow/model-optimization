<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfmot.sparsity.keras.strip_pruning" />
<meta itemprop="path" content="Stable" />
</div>

# tfmot.sparsity.keras.strip_pruning

Strip pruning wrappers from the model.

```python
tfmot.sparsity.keras.strip_pruning(model)
```

Defined in
[`python/core/sparsity/keras/prune.py`](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/sparsity/keras/prune.py).

<!-- Placeholder for "Used in" -->

Once a model has been pruned to required sparsity, this method can be used to
restore the original model with the sparse weights.

Only sequential and functional models are supported for now.

#### Arguments:

*   <b>`model`</b>: A `tf.keras.Model` instance with pruned layers.

#### Returns:

A keras model with pruning wrappers removed.

#### Raises:

*   <b>`ValueError`</b>: if the model is not a `tf.keras.Model` instance.
*   <b>`NotImplementedError`</b>: if the model is a subclass model.

Usage:

```python
orig_model = tf.keras.Model(inputs, outputs)
pruned_model = prune_low_magnitude(orig_model)
exported_model = strip_pruning(pruned_model)
```

The exported_model and the orig_model share the same structure.
