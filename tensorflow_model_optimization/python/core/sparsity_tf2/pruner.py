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
"""Helper functions to add support for magnitude-based model pruning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.summary import summary as summary_ops_v1
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched


class Pruner(object):
  """Implementation of magnitude-based weight pruning."""

  def __init__(self,
      pruning_schedule=pruning_sched.ConstantSparsity(0.5, 0),
      block_size=(1, 1),
      block_pooling_type='AVG',
  ):
    """The logic for magnitude-based pruning weight tensors.

    Args:
      pruning_schedule: A `PruningSchedule` object that controls pruning rate
        throughout training.
      block_size: The dimensions (height, weight) for the block sparse pattern
        in rank-2 weight tensors.
      block_pooling_type: (optional) The function to use to pool weights in the
        block. Must be 'AVG' or 'MAX'.
    """
    self._pruning_schedule = pruning_schedule
    self._block_size = list(block_size)
    self._block_pooling_type = block_pooling_type

  def _validate_block(self, pruning_vars):
    if self._block_size != [1, 1]:
      for weight, _, _ in pruning_vars:
        if weight.get_shape().ndims != 2:
          raise ValueError('Block Sparsity can only be used for layers which '
                           'have 2-dimensional weights.')

  def _update_mask(self, step, weights):
    """Updates the mask for a given weight tensor.

    This functions first estimates the threshold value such that
    a given fraction of weights have magnitude less than
    the threshold.

    Args:
      weights: The weight tensor that needs to be masked.

    Returns:
      new_threshold: The new value of the threshold based on weights, and
        sparsity at the current global_step
      new_mask: A numpy array of the same size and shape as weights containing
        0 or 1 to indicate which of the values in weights falls below
        the threshold

    Raises:
      ValueError: if sparsity is not defined
    """
    sparsity = self._pruning_schedule(step)[1]
    print("sparsity", sparsity)
    with tf.name_scope('pruning_ops'):
      abs_weights = tf.math.abs(weights)
      k = tf.dtypes.cast(
          tf.math.round(
              tf.dtypes.cast(tf.size(abs_weights), tf.float32) *
              (1 - sparsity)), tf.int32)
      # Sort the entire array
      values, _ = tf.math.top_k(
          tf.reshape(abs_weights, [-1]), k=tf.size(abs_weights))
      # Grab the (k-1)th value

      current_threshold = tf.gather(values, k - 1)
      new_mask = tf.dtypes.cast(
          tf.math.greater_equal(abs_weights, current_threshold), weights.dtype)
      print(f"updated new mask: {new_mask} from current threshold {current_threshold}")
    return current_threshold, new_mask

  def _maybe_update_block_mask(self, step, weights):
    """Performs block-granular masking of the weights.

    Block pruning occurs only if the block_height or block_width is > 1 and
    if the weight tensor, when squeezed, has ndims = 2. Otherwise, elementwise
    pruning occurs.
    Args:
      weights: The weight tensor that needs to be masked.

    Returns:
      new_threshold: The new value of the threshold based on weights, and
        sparsity at the current global_step
      new_mask: A numpy array of the same size and shape as weights containing
        0 or 1 to indicate which of the values in weights falls below
        the threshold

    Raises:
      ValueError: if block pooling function is not AVG or MAX
    """
    if self._block_size == [1, 1]:
      return self._update_mask(step, weights)

    abs_weights = tf.math.abs(weights)
    pooled_weights = pruning_utils.factorized_pool(
        abs_weights,
        window_shape=self._block_size,
        pooling_type=self._block_pooling_type,
        strides=self._block_size,
        padding='SAME')

    if pooled_weights.get_shape().ndims != 2:
      pooled_weights = tf.squeeze(pooled_weights)

    new_threshold, new_mask = self._update_mask(step, pooled_weights)

    updated_mask = pruning_utils.expand_tensor(new_mask, self._block_size)
    sliced_mask = tf.slice(
        updated_mask, [0, 0],
        [weights.get_shape()[0],
         weights.get_shape()[1]])
    return new_threshold, tf.reshape(sliced_mask, tf.shape(weights))

  def update_masks(self, pruning_vars, step):
    """Updates masks as per the pruning schedule.

    Args:
      pruning_vars: A list of (weight, mask, threshold) tuples
    """
    # TODO(xwinxu): depending on exact mid-level APIs, choose a way
    # to just validate just once?

    self._validate_block(pruning_vars)
    # TODO(Kaftan/xwinxu): verify if we needed the distribution strategy
    # logic from the tf1 code. (Tomer is dubious that we need in in tf2 loops)
    if self._pruning_schedule(step)[0]:
      for weight, mask, threshold in pruning_vars:
        new_threshold, new_mask = self._maybe_update_block_mask(step, weight)
        threshold.assign(new_threshold)
        mask.assign(new_mask)

  def add_pruning_summaries(self, step, pruning_vars):
    """Adds summaries of weight sparsities and thresholds."""
    # b/(139939526): update to use public API.
    summary = summary_ops_v1
    if tf.executing_eagerly():
      summary = summary_ops_v2
    summary.scalar('sparsity', self._pruning_schedule(step)[1])
    for _, mask, threshold in pruning_vars:
      summary.scalar(mask.name + '/sparsity', 1.0 - tf.math.reduce_mean(mask))
      summary.scalar(threshold.name + '/threshold', threshold)

  def _apply_mask(self, weight, mask):
    """Directly masks the weights (updating the weight variables)."""

    # TODO(Kaftan/xwinxu): figure out if this is totally unneeded now
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
      masked_weight = tf.math.multiply(weight, mask)
      values_and_vars.append((masked_weight, weight))
      if values_and_vars:
        tf.distribute.get_replica_context().merge_call(
            update_fn, args=(values_and_vars,))
    else:
      masked_weight = tf.math.multiply(weight, mask)
      weight.assign(masked_weight)

  def create_slots(self, optimizer, var):
    optimizer.add_slot(var, 'mask', initializer='ones')
    optimizer.add_slot(var, 'threshold', initializer=tf.zeros(shape=()))

  def prune(self, optimizer, var, grad):
    # Gradient is unused for low magnitude pruning
    mask = optimizer.get_slot(var, 'mask')
    threshold = optimizer.get_slot(var, 'threshold')
    self.update_masks([(var, mask, threshold)], step=optimizer.iterations)
    self._apply_mask(var, mask)

