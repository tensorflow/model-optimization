<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfmot.sparsity.keras.PruningSummaries" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="on_batch_begin"/>
<meta itemprop="property" content="on_batch_end"/>
<meta itemprop="property" content="on_epoch_begin"/>
<meta itemprop="property" content="on_epoch_end"/>
<meta itemprop="property" content="on_predict_batch_begin"/>
<meta itemprop="property" content="on_predict_batch_end"/>
<meta itemprop="property" content="on_predict_begin"/>
<meta itemprop="property" content="on_predict_end"/>
<meta itemprop="property" content="on_test_batch_begin"/>
<meta itemprop="property" content="on_test_batch_end"/>
<meta itemprop="property" content="on_test_begin"/>
<meta itemprop="property" content="on_test_end"/>
<meta itemprop="property" content="on_train_batch_begin"/>
<meta itemprop="property" content="on_train_batch_end"/>
<meta itemprop="property" content="on_train_begin"/>
<meta itemprop="property" content="on_train_end"/>
<meta itemprop="property" content="set_model"/>
<meta itemprop="property" content="set_params"/>
</div>

# tfmot.sparsity.keras.PruningSummaries

## Class `PruningSummaries`

A Keras callback for adding pruning summaries to tensorboard.

Defined in
[`python/core/sparsity/keras/pruning_callbacks.py`](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/sparsity/keras/pruning_callbacks.py).

<!-- Placeholder for "Used in" -->

Logs the sparsity(%) and threshold at a given iteration step.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    log_dir,
    update_freq='epoch',
    **kwargs
)
```

## Methods

<h3 id="on_batch_begin"><code>on_batch_begin</code></h3>

```python
on_batch_begin(
    batch,
    logs=None
)
```

A backwards compatibility alias for `on_train_batch_begin`.

<h3 id="on_batch_end"><code>on_batch_end</code></h3>

```python
on_batch_end(
    batch,
    logs=None
)
```

Writes scalar summaries for metrics on every training batch.

Performs profiling if current batch is in profiler_batches.

#### Arguments:

*   <b>`batch`</b>: Integer, index of batch within the current epoch.
*   <b>`logs`</b>: Dict. Metric results for this batch.

<h3 id="on_epoch_begin"><code>on_epoch_begin</code></h3>

```python
on_epoch_begin(
    epoch,
    logs=None
)
```

Called at the start of an epoch.

Subclasses should override for any actions to run. This function should only be
called during TRAIN mode.

#### Arguments:

*   <b>`epoch`</b>: integer, index of epoch.
*   <b>`logs`</b>: dict. Currently no data is passed to this argument for this
    method but that may change in the future.

<h3 id="on_epoch_end"><code>on_epoch_end</code></h3>

```python
on_epoch_end(
    batch,
    logs=None
)
```

<h3 id="on_predict_batch_begin"><code>on_predict_batch_begin</code></h3>

```python
on_predict_batch_begin(
    batch,
    logs=None
)
```

Called at the beginning of a batch in `predict` methods.

Subclasses should override for any actions to run.

#### Arguments:

*   <b>`batch`</b>: integer, index of batch within the current epoch.
*   <b>`logs`</b>: dict. Has keys `batch` and `size` representing the current
    batch number and the size of the batch.

<h3 id="on_predict_batch_end"><code>on_predict_batch_end</code></h3>

```python
on_predict_batch_end(
    batch,
    logs=None
)
```

Called at the end of a batch in `predict` methods.

Subclasses should override for any actions to run.

#### Arguments:

*   <b>`batch`</b>: integer, index of batch within the current epoch.
*   <b>`logs`</b>: dict. Metric results for this batch.

<h3 id="on_predict_begin"><code>on_predict_begin</code></h3>

```python
on_predict_begin(logs=None)
```

Called at the beginning of prediction.

Subclasses should override for any actions to run.

#### Arguments:

*   <b>`logs`</b>: dict. Currently no data is passed to this argument for this
    method but that may change in the future.

<h3 id="on_predict_end"><code>on_predict_end</code></h3>

```python
on_predict_end(logs=None)
```

Called at the end of prediction.

Subclasses should override for any actions to run.

#### Arguments:

*   <b>`logs`</b>: dict. Currently no data is passed to this argument for this
    method but that may change in the future.

<h3 id="on_test_batch_begin"><code>on_test_batch_begin</code></h3>

```python
on_test_batch_begin(
    batch,
    logs=None
)
```

Called at the beginning of a batch in `evaluate` methods.

Also called at the beginning of a validation batch in the `fit` methods, if
validation data is provided.

Subclasses should override for any actions to run.

#### Arguments:

*   <b>`batch`</b>: integer, index of batch within the current epoch.
*   <b>`logs`</b>: dict. Has keys `batch` and `size` representing the current
    batch number and the size of the batch.

<h3 id="on_test_batch_end"><code>on_test_batch_end</code></h3>

```python
on_test_batch_end(
    batch,
    logs=None
)
```

Called at the end of a batch in `evaluate` methods.

Also called at the end of a validation batch in the `fit` methods, if validation
data is provided.

Subclasses should override for any actions to run.

#### Arguments:

*   <b>`batch`</b>: integer, index of batch within the current epoch.
*   <b>`logs`</b>: dict. Metric results for this batch.

<h3 id="on_test_begin"><code>on_test_begin</code></h3>

```python
on_test_begin(logs=None)
```

Called at the beginning of evaluation or validation.

Subclasses should override for any actions to run.

#### Arguments:

*   <b>`logs`</b>: dict. Currently no data is passed to this argument for this
    method but that may change in the future.

<h3 id="on_test_end"><code>on_test_end</code></h3>

```python
on_test_end(logs=None)
```

Called at the end of evaluation or validation.

Subclasses should override for any actions to run.

#### Arguments:

*   <b>`logs`</b>: dict. Currently no data is passed to this argument for this
    method but that may change in the future.

<h3 id="on_train_batch_begin"><code>on_train_batch_begin</code></h3>

```python
on_train_batch_begin(
    batch,
    logs=None
)
```

Called at the beginning of a training batch in `fit` methods.

Subclasses should override for any actions to run.

#### Arguments:

*   <b>`batch`</b>: integer, index of batch within the current epoch.
*   <b>`logs`</b>: dict. Has keys `batch` and `size` representing the current
    batch number and the size of the batch.

<h3 id="on_train_batch_end"><code>on_train_batch_end</code></h3>

```python
on_train_batch_end(
    batch,
    logs=None
)
```

Called at the end of a training batch in `fit` methods.

Subclasses should override for any actions to run.

#### Arguments:

*   <b>`batch`</b>: integer, index of batch within the current epoch.
*   <b>`logs`</b>: dict. Metric results for this batch.

<h3 id="on_train_begin"><code>on_train_begin</code></h3>

```python
on_train_begin(logs=None)
```

<h3 id="on_train_end"><code>on_train_end</code></h3>

```python
on_train_end(logs=None)
```

<h3 id="set_model"><code>set_model</code></h3>

```python
set_model(model)
```

Sets Keras model and writes graph if specified.

<h3 id="set_params"><code>set_params</code></h3>

```python
set_params(params)
```
