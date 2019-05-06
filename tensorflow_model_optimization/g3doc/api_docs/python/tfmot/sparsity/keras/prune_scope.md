<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfmot.sparsity.keras.prune_scope" />
<meta itemprop="path" content="Stable" />
</div>

# tfmot.sparsity.keras.prune_scope

Provides a scope in which Pruned layers and models can be deserialized.

```python
tfmot.sparsity.keras.prune_scope()
```

Defined in
[`python/core/sparsity/keras/prune.py`](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/sparsity/keras/prune.py).

<!-- Placeholder for "Used in" -->

If a keras model or layer has been pruned, it needs to be within this scope to
be successfully deserialized.

#### Returns:

    Object of type `CustomObjectScope` with pruning objects included.

Example:

```python
pruned_model = prune_low_magnitude(model, **self.params)
keras.models.save_model(pruned_model, keras_file)

with prune_scope():
  loaded_model = keras.models.load_model(keras_file)
```
