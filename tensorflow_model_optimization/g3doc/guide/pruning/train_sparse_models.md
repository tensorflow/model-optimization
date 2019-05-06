# Train sparse TensorFlow models with Keras

This document describes the Keras based API that implements magnitude-based
pruning of neural network's weight tensors.

Weight pruning means eliminating unnecessary values in the weight tensors. We
set the neural network parameters' values to zero to remove what we estimate are
unnecessary connections between the layers of a neural network. This is done
during the training process to allow the neural network to adapt to the changes.

Our Keras-based weight pruning API uses a straightforward, yet broadly
applicable magnitude-based pruning [algorithm](https://arxiv.org/abs/1710.01878)
designed to iteratively remove connections based on their magnitude during
training. Fundamentally, a final target sparsity is specified (e.g. 90%), along
with a schedule to perform the pruning (e.g. start pruning at step 2,000, stop
at step 10,000, and do it every 100 steps), and an optional configuration for
the pruning structure (e.g. apply to individual values or blocks of values in
certain shape).

As training proceeds, the pruning routine will be scheduled to execute,
eliminating (i.e. setting to zero) the weights with the lowest magnitude values
(i.e. those closest to zero) until the current sparsity target is reached. Every
time the pruning routine is scheduled to execute, the current sparsity target is
recalculated, starting from 0% until it reaches the final target sparsity at the
end of the pruning schedule by gradually increasing it according to a smooth
ramp-up function.

Just like the schedule, the ramp-up function can be tweaked as needed. For
example, in certain cases, it may be convenient to schedule the training
procedure to start after a certain step when some convergence level has been
achieved, or end pruning earlier than the total number of training steps in your
training program to further fine-tune the system at the final target sparsity
level.

In the following sections we describe in detail how to make use of the API.


## How to use the Keras API <a name="keras-api-usage"></a>

## Model creation <a name="model-creation"></a>

We provide a prune_low_magnitude() method which is able to take a keras layer, a
list of keras layers, or a keras model and apply the pruning wrapper
accordingly. You can use it when building the model, or with a pre-built one.

For example, to wrap the layers when building the model:

```python
model = tf.keras.Sequential([
    prune_low_magnitude(tf.keras.layers.Dense(10), **pruning_params, input_shape=input_shape),
    tf.keras.layers.Flatten()
])

# Compile the model as usual
model.compile(
    loss=...,
    optimizer=...,
    metrics=[...])
```

To prune a pre-built model:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense((10), input_shape=input_shape),
    tf.keras.layers.Flatten()
])
pruned_model = prune_low_magnitude(model, **pruning_params)

# Compile the model as usual
pruned_model.compile(
    loss=...,
    optimizer=...,
    metrics=[...])
```

Layers supported: all keras built-in layers. For custom layers, see the
[`Prune a custom layer`](#prune-a-custom-layer) section below for instructions.

Models supported: Sequential and Functional models, but not Subclass models.

## Train the pruned model

To train the pruned model, you need to use the following callbacks with the
model.fit() method:

```python
callbacks = [
    # Update the pruning step
    pruning_callbacks.UpdatePruningStep(),
    # Add summaries to keep track of the sparsity in different layers during training
    pruning_callbacks.PruningSummaries(log_dir=training_log_dir)
]

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=(x_test, y_test))
```

They're responsible for updating the pruning step during training, and writing
summaries of the pruning status like sparsity and threshold of pruned layers.

## Save/restore a checkpoint of the pruned model

If you want to save a checkpoint of the pruned model and reload it to continue
with training, you can use these standard keras API:

```python
saved_model = tf.keras.models.save_model(model, keras_file)
with prune_scope():
  loaded_model = keras.models.load_model(keras_file)

loaded_model.fit(...)
```

*   By default saved_model() sets include_optmizer to True. Please DO NOT change
    this if you want to reload the pruned model for training. We need to keep
    the optimizer state across training sessions for pruning to work properly.
*   The prune_scope() provides a custom object name scope to resolve the pruning
    wrapper class during deserialization.

## Removing pruning wrappers from the pruned model

Once the model is trained to reach the target sparsity level and a sastifactory
accuracy, it is necessary to remove the wrappers added to the model to finalize
pruning. This can be done with calling the strip_pruning() method as:

```python
# The exported model has the same architecture with the original non-pruned model. Only the weight tensors are pruned to be sparse tensors.
final_model = strip_pruning(pruned_model)
```

Then you can export the model for serving with:

```python
tf.keras.model.save_model(final_model, file, include_optimizer=False)
```

## Advanced usage patterns

### Prune a custom layer

The pruning wrapper can also be applied to a user-defined keras layer. Custom
layers can inherit from the PrunableLayer interface and implement the
get_prunable_weights() method to be pruned. Please refer to
[PrunableLayer](../api_docs/python/tfmot/sparsity/keras/PrunableLayer).

### Block sparsity

Configure this via
[prune_low_magnitude](../api_docs/python/tfmot/sparsity/keras/prune_low_magnitude).

For some hardware architectures, it may be beneficial to induce spatially
correlated sparsity. To train models in which the weight tensors have block
sparse structure, set the *block_size* parameter to the desired block
configuration (2x2, 4x4, 4x1, 1x8, etc). Currently, block sparsity is only
supported for weight tensors which can be squeezed to rank 2. The matrix is
partitioned into non-overlapping blocks and the either the average or max
absolute value in this block is taken as a proxy for the entire block (set by
*block_pooling_type* parameter). The convolution layer tensors are always pruned
used block dimensions of [1,1].
