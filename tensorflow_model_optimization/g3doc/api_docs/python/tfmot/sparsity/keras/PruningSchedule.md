<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfmot.sparsity.keras.PruningSchedule" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfmot.sparsity.keras.PruningSchedule

## Class `PruningSchedule`

Specifies when to prune layer and the sparsity(%) at each training step.

Defined in
[`python/core/sparsity/keras/pruning_schedule.py`](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/sparsity/keras/pruning_schedule.py).

<!-- Placeholder for "Used in" -->

PruningSchedule controls pruning during training by notifying at each step
whether the layer's weights should be pruned or not, and the sparsity(%) at
which they should be pruned.

It can be invoked as a `callable` by providing the training `step` Tensor. It
returns a tuple of bool and float tensors.

```python
  should_prune, sparsity = pruning_schedule(step)
```

You can inherit this class to write your own custom pruning schedule.

## Methods

<h3 id="__call__"><code>__call__</code></h3>

```python
__call__(step)
```

Returns the sparsity(%) to be applied.

If the returned sparsity(%) is 0, pruning is ignored for the step.

#### Args:

*   <b>`step`</b>: Current step in graph execution.

#### Returns:

Sparsity (%) that should be applied to the weights for the step.

<h3 id="from_config"><code>from_config</code></h3>

```python
@classmethod
from_config(
    cls,
    config
)
```

Instantiates a `PruningSchedule` from its config.

#### Args:

*   <b>`config`</b>: Output of `get_config()`.

#### Returns:

A `PruningSchedule` instance.

<h3 id="get_config"><code>get_config</code></h3>

```python
get_config()
```
