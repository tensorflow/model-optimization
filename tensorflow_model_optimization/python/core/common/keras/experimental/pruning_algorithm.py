# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Pruning algorithm based on low magnitude."""

import tensorflow.compat.v2 as tf

from tensorflow_model_optimization.python.core.common import algorithm
from tensorflow_model_optimization.python.core.common.keras import wrapper
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule


def prune_low_magnitude(to_prune, schedule):
  """Possible replacement of tfmot.sparsity.keras.prune_low_magnitude."""
  assert isinstance(to_prune, tf.keras.Model)

  config = PrunerConfig()

  def _add_pruning_wrapper(layer):
    return wrapper.wrap_keras_layer(layer, LowMagnitudePruner(schedule), config)

  return tf.keras.models.clone_model(
      to_prune, clone_function=_add_pruning_wrapper)


class LowMagnitudePruner(algorithm.OptimizationAlgorithmInterface):
  """A direct implementation of a pruning algorithm."""

  def __init__(self, schedule):
    if not isinstance(schedule, pruning_schedule.PruningSchedule):
      raise ValueError('pruning_schedule must be a `PruningSchedule` object.')
    self._schedule = schedule

  def initial_state(self, weights):
    trainable = {'weights': weights}
    non_trainable = {
        'masks': tf.nest.map_structure(tf.ones_like, weights),
        'step': tf.constant(0)
    }
    return algorithm.StateRepr(trainable, non_trainable)

  def get_weight_tensors(self, state):
    return tf.nest.map_structure(tf.math.multiply,
                                 state.trainable['weights'],
                                 state.non_trainable['masks'])

  def update_state(self, state):
    update_state_bool, sparsity = self._schedule(state.non_trainable['step'])
    masks, weights = tf.cond(
        update_state_bool,
        lambda: self._update_state(sparsity, state.trainable['weights']),
        lambda: (state.non_trainable['masks'], state.trainable['weights']))
    step = state.non_trainable['step'] + 1
    trainable = {'weights': weights}
    non_trainable = {'masks': masks, 'step': step}
    return algorithm.StateRepr(trainable, non_trainable)

  def _update_state(self, sparsity, weights):
    masks = tf.nest.map_structure(
        lambda w: _target_sparsity_to_mask(sparsity, w), weights)
    weights = tf.nest.map_structure(tf.math.multiply, weights, masks)
    return masks, weights


def _target_sparsity_to_mask(sparsity, weights):
  abs_weights = tf.abs(weights)
  target_k = tf.cast(
      tf.round(tf.cast(tf.size(abs_weights), tf.float32) * (1 - sparsity)),
      tf.int32)
  values, _ = tf.math.top_k(
      tf.reshape(abs_weights, [-1]), k=tf.size(abs_weights))
  threshold = tf.gather(values, target_k - 1)
  return tf.cast(tf.math.greater_equal(abs_weights, threshold), tf.float32)


class PrunerConfig(wrapper.KerasOptimizationAlgorithmConfig):

  def weight_attrs_to_optimize(self, layer_cls):
    return _PRUNE_WEIGHTS_MAP.get(layer_cls, [])


_PRUNE_WEIGHTS_MAP = {
    tf.keras.layers.Dense: ['kernel'],
}
