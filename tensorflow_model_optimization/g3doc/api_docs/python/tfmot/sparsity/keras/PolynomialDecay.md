<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfmot.sparsity.keras.PolynomialDecay" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfmot.sparsity.keras.PolynomialDecay

## Class `PolynomialDecay`

Pruning Schedule with a PolynomialDecay function.

Inherits From:
[`PruningSchedule`](../../../tfmot/sparsity/keras/PruningSchedule.md)

Defined in
[`python/core/sparsity/keras/pruning_schedule.py`](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/sparsity/keras/pruning_schedule.py).

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    initial_sparsity,
    final_sparsity,
    begin_step,
    end_step,
    power=3,
    frequency=100
)
```

Initializes a Pruning schedule with a PolynomialDecay function.

Pruning rate grows rapidly in the beginning from initial_sparsity, but then
plateaus slowly to the target sparsity. The function applied is

current_sparsity = final_sparsity + (initial_sparsity - final_sparsity) * (1 -
(step - begin_step)/(end_step - begin_step)) ^ exponent

which is a polynomial decay function. See
[paper](https://arxiv.org/abs/1710.01878).

#### Args:

*   <b>`initial_sparsity`</b>: Sparsity (%) at which pruning begins.
*   <b>`final_sparsity`</b>: Sparsity (%) at which pruning ends.
*   <b>`begin_step`</b>: Step at which to begin pruning.
*   <b>`end_step`</b>: Step at which to end pruning.
*   <b>`power`</b>: Exponent to be used in the sparsity function.
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
