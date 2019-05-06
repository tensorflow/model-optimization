<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfmot.sparsity.keras.ConstantSparsity" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfmot.sparsity.keras.ConstantSparsity

## Class `ConstantSparsity`

Pruning schedule with constant sparsity(%) throughout training.

Inherits From:
[`PruningSchedule`](../../../tfmot/sparsity/keras/PruningSchedule.md)

Defined in
[`python/core/sparsity/keras/pruning_schedule.py`](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/sparsity/keras/pruning_schedule.py).

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    target_sparsity,
    begin_step,
    end_step=-1,
    frequency=100
)
```

Initializes a Pruning schedule with constant sparsity.

Sparsity is applied in the interval [`begin_step`, `end_step`] every `frequency`
steps. At each applicable step, the sparsity(%) is constant.

#### Args:

*   <b>`target_sparsity`</b>: A scalar float representing the target sparsity
    value.
*   <b>`begin_step`</b>: Step at which to begin pruning.
*   <b>`end_step`</b>: Step at which to end pruning. `-1` by default. `-1`
    implies continuing to prune till the end of training.
*   <b>`frequency`</b>: Only apply pruning every `frequency` steps.

## Methods

<h3 id="__call__"><code>__call__</code></h3>

```python
__call__(step)
```

<h3 id="from_config"><code>from_config</code></h3>

```python
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
