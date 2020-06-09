# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=protected-access,missing-docstring,unused-argument
"""Entry point for pruning models during training."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from tensorflow_model_optimization.python.core.sparsity_tf2 import pruning_wrapper

keras = tf.keras
custom_object_scope = tf.keras.utils.custom_object_scope


def prune_scope():
  """Provides a scope in which Pruned layers and models can be deserialized.

  For TF 2.X: this is not needed for SavedModel or TF checkpoints, which are
  the recommended serialization formats.

  For TF 1.X: if a tf.keras h5 model or layer has been pruned, it needs to be
  within this
  scope to be successfully deserialized. This is not needed for loading just
  keras weights.

  Returns:
      Object of type `CustomObjectScope` with pruning objects included.

  Example:

  ```python
  pruned_model = prune_low_magnitude(model, **self.params)
  keras.models.save_model(pruned_model, keras_file)

  with prune_scope():
    loaded_model = keras.models.load_model(keras_file)
  ```
  """
  return custom_object_scope(
      {'PruneLowMagnitude': pruning_wrapper.PrunableWrapper})


def prune_low_magnitude(to_prune,
                        pruning_schedule=pruning_sched.ConstantSparsity(0.5, 0),
                        block_size=(1, 1),
                        block_pooling_type='AVG',
                        **kwargs):
  """Modify a tf.keras layer or model to be pruned during training.

  This function wraps a tf.keras model or layer with pruning functionality which
  sparsifies the layer's weights during training. For example, using this with
  50% sparsity will ensure that 50% of the layer's weights are zero.

  The function accepts either a single keras layer
  (subclass of `tf.keras.layers.Layer`), list of keras layers or a Sequential
  or Functional tf.keras model and handles them appropriately.

  If it encounters a layer it does not know how to handle, it will throw an
  error. While pruning an entire model, even a single unknown layer would lead
  to an error.

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

  Pretrained models: you must first load the weights and then apply the
  prune API:

  ```python
  model.load_weights(...)
  model = prune_low_magnitude(model)
  ```

  Optimizer: this function removes the optimizer. The user is expected to
  compile the model
  again. It's easiest to rely on the default (step starts at 0) and then
  use that to determine the desired begin_step for the pruning_schedules.

  Checkpointing: checkpointing should include the optimizer, not just the
  weights. Pruning supports
  checkpointing though
  upon inspection, the weights of checkpoints are not sparse
  (https://github.com/tensorflow/model-optimization/issues/206).

  Arguments:
      to_prune: A single keras layer, list of keras layers, or a
        `tf.keras.Model` instance.
      pruning_schedule: A `PruningSchedule` object that controls pruning rate
        throughout training.
      block_size: (optional) The dimensions (height, weight) for the block
        sparse pattern in rank-2 weight tensors.
      block_pooling_type: (optional) The function to use to pool weights in the
        block. Must be 'AVG' or 'MAX'.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
        Ignored when to_prune is not a keras layer.

  Returns:
    Layer or model modified with pruning wrappers. Optimizer is removed.

  Raises:
    ValueError: if the keras layer is unsupported, or the keras model contains
    an unsupported layer.
  """

  def _prune_list(layers):
    wrapped_layers = []

    for layer in layers:
      # Allow layer that is already wrapped by the pruning wrapper
      # to be used as is.
      # No need to wrap the input layer either.
      if isinstance(layer, pruning_wrapper.PrunableWrapper):
        wrapped_layers.append(layer)
      elif isinstance(layer, keras.layers.InputLayer):
        # TODO(yunluli): Replace with a clone function in keras.
        wrapped_layers.append(layer.__class__.from_config(layer.get_config()))
      else:
        wrapped_layers.append(
            pruning_wrapper.PrunableWrapper(layer))

    return wrapped_layers

  def _add_pruning_wrapper(layer):
    if isinstance(layer, pruning_wrapper.PrunableWrapper):
      return layer
    return pruning_wrapper.PrunableWrapper(layer)

  params = {
      'pruning_schedule': pruning_schedule,
      'block_size': block_size,
      'block_pooling_type': block_pooling_type
  }
  is_sequential_or_functional = isinstance(
      to_prune, keras.Model) and (isinstance(to_prune, keras.Sequential) or
                                  to_prune._is_graph_network)

  # A subclassed model is also a subclass of keras.layers.Layer.
  is_keras_layer = isinstance(
      to_prune, keras.layers.Layer) and not isinstance(to_prune, keras.Model)

  if isinstance(to_prune, list):
    return _prune_list(to_prune)
  elif is_sequential_or_functional:
    return keras.models.clone_model(
        to_prune, input_tensors=None, clone_function=_add_pruning_wrapper)
  elif is_keras_layer:
    params.update(kwargs)
    return pruning_wrapper.PrunableWrapper(to_prune)
  else:
    raise ValueError(
        '`prune_low_magnitude` can only prune an object of the following '
        'types: tf.keras.models.Sequential, tf.keras functional model, '
        'tf.keras.layers.Layer, list of tf.keras.layers.Layer. You passed '
        'an object of type: {input}.'.format(input=to_prune.__class__.__name__))
