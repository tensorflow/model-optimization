<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfmot.sparsity.keras.PrunableLayer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="get_prunable_weights"/>
</div>

# tfmot.sparsity.keras.PrunableLayer

## Class `PrunableLayer`

Abstract Base Class for making your own keras layer prunable.

Defined in
[`python/core/sparsity/keras/prunable_layer.py`](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/sparsity/keras/prunable_layer.py).

<!-- Placeholder for "Used in" -->

Custom keras layers which want to add pruning should implement this class.

## Methods

<h3 id="get_prunable_weights"><code>get_prunable_weights</code></h3>

```python
get_prunable_weights()
```

Returns list of prunable weight tensors.

All the weight tensors which the layer wants to be pruned during training must
be returned by this method.

Returns: List of weight tensors/kernels in the keras layer which must be pruned
during training.
