<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfmot.sparsity.keras.prune_low_magnitude" />
<meta itemprop="path" content="Stable" />
</div>

# tfmot.sparsity.keras.prune_low_magnitude

Modify a keras layer or model to be pruned during training.

```python
tfmot.sparsity.keras.prune_low_magnitude(
    to_prune,
    pruning_schedule=pruning_sched.ConstantSparsity(0.5, 0),
    block_size=(1, 1),
    block_pooling_type='AVG',
    **kwargs
)
```

Defined in
[`python/core/sparsity/keras/prune.py`](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/sparsity/keras/prune.py).

<!-- Placeholder for "Used in" -->

This function wraps a keras model or layer with pruning functionality which
sparsifies the layer's weights during training. For example, using this with 50%
sparsity will ensure that 50% of the layer's weights are zero.

The function accepts either a single keras layer (subclass of
`keras.layers.Layer`), list of keras layers or a keras model (instance of
`keras.models.Model`) and handles them appropriately.

If it encounters a layer it does not know how to handle, it will throw an error.
While pruning an entire model, even a single unknown layer would lead to an
error.

Prune a model:

```python
pruning_params = {
    'pruning_schedule': ConstantSparsity(0.5, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}

model = prune_low_magnitude(
    keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(100,)),
        layers.Dense(2, activation='sigmoid')
    ]), **pruning_params)
```

Prune a layer:

```python
pruning_params = {
    'pruning_schedule': PolynomialDecay(initial_sparsity=0.2,
        final_sparsity=0.8, begin_step=1000, end_step=2000),
    'block_size': (2, 3),
    'block_pooling_type': 'MAX'
}

model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(100,)),
    prune_low_magnitude(layers.Dense(2, activation='tanh'), **pruning_params)
])
```

#### Arguments:

*   <b>`to_prune`</b>: A single keras layer, list of keras layers, or a
    `tf.keras.Model` instance.
*   <b>`pruning_schedule`</b>: A `PruningSchedule` object that controls pruning
    rate throughout training.
*   <b>`block_size`</b>: (optional) The dimensions (height, weight) for the
    block sparse pattern in rank-2 weight tensors.
*   <b>`block_pooling_type`</b>: (optional) The function to use to pool weights
    in the block. Must be 'AVG' or 'MAX'.
*   <b>`**kwargs`</b>: Additional keyword arguments to be passed to the keras
    layer. Ignored when to_prune is not a keras layer.

#### Returns:

Layer or model modified with pruning wrappers.

#### Raises:

*   <b>`ValueError`</b>: if the keras layer is unsupported, or the keras model
    contains an unsupported layer.
