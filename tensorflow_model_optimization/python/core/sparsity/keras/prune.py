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

from tensorflow.python import keras
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.utils.generic_utils import custom_object_scope
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper


def prune_scope():
  """Provides a scope in which Pruned layers and models can be deserialized.

  If a keras model or layer has been pruned, it needs to be within this scope
  to be successfully deserialized.

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
      {'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude})


def prune_low_magnitude(to_prune,
                        pruning_schedule=pruning_sched.ConstantSparsity(0.5, 0),
                        block_size=(1, 1),
                        block_pooling_type='AVG',
                        **kwargs):
  """Modify a keras layer or model to be pruned during training.

  This function wraps a keras model or layer with pruning functionality which
  sparsifies the layer's weights during training. For example, using this with
  50% sparsity will ensure that 50% of the layer's weights are zero.

  The function accepts either a single keras layer
  (subclass of `keras.layers.Layer`), list of keras layers or a keras model
  (instance of `keras.models.Model`) and handles them appropriately.

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
    Layer or model modified with pruning wrappers.

  Raises:
    ValueError: if the keras layer is unsupported, or the keras model contains
    an unsupported layer.
  """

  def _prune_list(layers, **params):
    wrapped_layers = []

    for layer in layers:
      # Allow layer that is already wrapped by the pruning wrapper
      # to be used as is.
      # No need to wrap the input layer either.
      if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        wrapped_layers.append(layer)
      elif isinstance(layer, InputLayer):
        # TODO(yunluli): Replace with a clone function in keras.
        wrapped_layers.append(layer.__class__.from_config(layer.get_config()))
      else:
        wrapped_layers.append(
            pruning_wrapper.PruneLowMagnitude(layer, **params))

    return wrapped_layers

  def _add_pruning_wrapper(layer):
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      return layer
    return pruning_wrapper.PruneLowMagnitude(layer, **params)

  params = {
      'pruning_schedule': pruning_schedule,
      'block_size': block_size,
      'block_pooling_type': block_pooling_type
  }

  if isinstance(to_prune, list):
    return _prune_list(to_prune, **params)
  elif isinstance(to_prune, keras.Model):
    return keras.models.clone_model(
        to_prune, input_tensors=None, clone_function=_add_pruning_wrapper)
  elif isinstance(to_prune, keras.layers.Layer):
    params.update(kwargs)
    return pruning_wrapper.PruneLowMagnitude(to_prune, **params)


def strip_pruning(model):
  """Strip pruning wrappers from the model.

  Once a model has been pruned to required sparsity, this method can be used
  to restore the original model with the sparse weights.

  Only sequential and functional models are supported for now.

  Arguments:
      model: A `tf.keras.Model` instance with pruned layers.

  Returns:
    A keras model with pruning wrappers removed.

  Raises:
    ValueError: if the model is not a `tf.keras.Model` instance.
    NotImplementedError: if the model is a subclass model.

  Usage:

  ```python
  orig_model = tf.keras.Model(inputs, outputs)
  pruned_model = prune_low_magnitude(orig_model)
  exported_model = strip_pruning(pruned_model)
  ```
  The exported_model and the orig_model share the same structure.
  """

  if not isinstance(model, keras.Model):
    raise ValueError(
        'Expected model to be a `tf.keras.Model` instance but got: ', model)

  def _strip_pruning_wrapper(layer):
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      # The _batch_input_shape attribute in the first layer makes a Sequential
      # model to be built. This makes sure that when we remove the wrapper from
      # the first layer the model's built state preserves.
      if not hasattr(layer.layer, '_batch_input_shape') and hasattr(
          layer, '_batch_input_shape'):
        layer.layer._batch_input_shape = layer._batch_input_shape
      return layer.layer
    return layer

  return keras.models.clone_model(
      model, input_tensors=None, clone_function=_strip_pruning_wrapper)
