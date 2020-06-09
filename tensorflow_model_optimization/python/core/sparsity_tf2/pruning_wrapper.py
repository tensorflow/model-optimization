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
"""A Keras wrapper to add pruning related variables to a layer."""

# pylint: disable=missing-docstring,g-multiple-import,unused-import,protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
# import g3
import numpy as np
import tensorflow as tf

# b/(139939526): update to use public API.
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils

from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry

keras = tf.keras
Wrapper = keras.layers.Wrapper


class PrunableWrapper(Wrapper):
  """This wrapper augments a keras layer so the weight tensor may be pruned.

  Custom keras layers:
  The pruning wrapper can also be applied to a user-defined keras layer.
  Such a layer may contain one or more weight tensors that may be pruned.
  To apply pruning wrapper to such layers, set prunable_weight_names to mark
  the weight tensors for pruning.
  """

  def __init__(self,
               layer,
               **kwargs):
    """Create a pruning wrapper for a keras layer.

    Args:
      layer: The keras layer to be pruned.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    # An instance of the Pruning class. This class contains the logic to prune
    # the weights of this layer.
    self.pruning_obj = None

    # A list of all (weight,mask,threshold) tuples for this layer
    self.pruning_vars = []

    if not isinstance(layer, tf.keras.layers.Layer):
      raise ValueError(
          'Please initialize `Prune` layer with a '
          '`Layer` instance. You passed: {input}'.format(input=layer))

    # TODO(pulkitb): This should be pushed up to the wrappers.py
    # Name the layer using the wrapper and underlying layer name.
    # Prune(Dense) becomes prune_dense_1
    kwargs.update({'name': '{}_{}'.format(
        generic_utils.to_snake_case(self.__class__.__name__), layer.name)})

    if isinstance(layer, prunable_layer.PrunableLayer):
      # Custom layer in client code which supports pruning.
      super(Wrapper, self).__init__(layer, **kwargs)
    elif prune_registry.PruneRegistry.supports(layer):
      # Built-in keras layers which support pruning.
      super(Wrapper, self).__init__(
          prune_registry.PruneRegistry.make_prunable(layer), **kwargs)
    else:
      raise ValueError(
          'Please initialize `Prune` with a supported layer. Layers should '
          'either be a `PrunableLayer` instance, or should be supported by the '
          'PruneRegistry. You passed: {input}'.format(input=layer.__class__))

    # TODO(yunluli): Work-around to handle the first layer of Sequential model
    # properly. Can remove this when it is implemented in the Wrapper base
    # class.
    #
    # Enables end-user to prune the first layer in Sequential models, while
    # passing the input shape to the original layer.
    #
    # tf.keras.Sequential(
    #   prune_low_magnitude(tf.keras.layers.Dense(2, input_shape=(3,)))
    # )
    #
    # as opposed to
    #
    # tf.keras.Sequential(
    #   prune_low_magnitude(tf.keras.layers.Dense(2), input_shape=(3,))
    # )
    #
    # Without this code, the pruning wrapper doesn't have an input
    # shape and being the first layer, this causes the model to not be
    # built. Being not built is confusing since the end-user has passed an
    # input shape.
    if not hasattr(self, '_batch_input_shape') and hasattr(
        layer, '_batch_input_shape'):
      self._batch_input_shape = self.layer._batch_input_shape

  def build(self, input_shape):
    super(PrunableWrapper, self).build(input_shape)

    weight_vars, mask_vars, threshold_vars = [], [], []

    self.prunable_weights = self.layer.get_prunable_weights()

    # For each of the prunable weights, add mask and threshold variables
    for weight in self.prunable_weights:
      mask = self.add_variable(
          'mask',
          shape=weight.shape,
          # TODO(xwinxu): This will need to be generalized to support
          #  custom initializers
          initializer=tf.keras.initializers.get('ones'),
          dtype=weight.dtype,
          trainable=False,
          aggregation=tf.VariableAggregation.MEAN)
      threshold = self.add_variable(
          'threshold',
          shape=[],
          initializer=tf.keras.initializers.get('zeros'),
          dtype=weight.dtype,
          trainable=False,
          aggregation=tf.VariableAggregation.MEAN)

      weight_vars.append(weight)
      mask_vars.append(mask)
      threshold_vars.append(threshold)
    self.pruning_vars = list(zip(weight_vars, mask_vars, threshold_vars))

  def call(self, *args, **kwargs):
    self._mask_weights()
    return self.layer(*args, **kwargs)


  def _mask_weights(self):
    """Directly masks the weights (updating the weight variables)."""

    def update_fn(distribution, values_and_vars):
      # TODO(yunluli): Need this ReduceOp because the weight is created by the
      # layer wrapped, so we don't have control of its aggregation policy. May
      # be able to optimize this when distribution strategy supports easier
      # update to mirrored variables in replica context.
      reduced_values = distribution.extended.batch_reduce_to(
          tf.distribute.ReduceOp.MEAN, values_and_vars)
      var_list = [v for _, v in values_and_vars]
      values_and_vars = zip(reduced_values, var_list)

      def update_var(variable, reduced_value):
        return variable.assign(reduced_value)

      update_objs = []
      for value, var in values_and_vars:
        update_objs.append(
            distribution.extended.update(var, update_var, args=(value,)))

      return tf.group(update_objs)

    if tf.distribute.get_replica_context():
      values_and_vars = []
      for weight, mask, _ in self._pruning_vars:
        masked_weight = tf.math.multiply(weight, mask)
        values_and_vars.append((masked_weight, weight))
      if values_and_vars:
        tf.distribute.get_replica_context().merge_call(
            update_fn, args=(values_and_vars,))
    else:
      for weight, mask, _ in self._pruning_vars:
        masked_weight = tf.math.multiply(weight, mask)
        weight.assign(masked_weight)

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(input_shape)

  def get_config(self):
    base_config = super(PrunableWrapper, self).get_config()
    config = { }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()

    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    layer = deserialize_layer(config.pop('layer'))
    config['layer'] = layer

    return cls(**config)
