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

from tensorflow_model_optimization.python.core.keras import compat as tf_compat
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils


class Pruning(object):
  """Implementation of magnitude-based weight pruning."""

  def __init__(self,
               training_step_fn,
               pruning_vars,
               pruning_schedule,
               block_size,
               block_pooling_type,
               sparsity_m_by_n=None):
    """The logic for magnitude-based pruning weight tensors.

    Args:
      training_step_fn: A callable that returns the training step.
      pruning_vars: A list of (weight, mask, threshold) tuples
      pruning_schedule: A `PruningSchedule` object that controls pruning rate
        throughout training.
      block_size: The dimensions (height, weight) for the block sparse pattern
        in rank-2 weight tensors.
      block_pooling_type: (optional) The function to use to pool weights in the
        block. Must be 'AVG' or 'MAX'.
      sparsity_m_by_n: default None, otherwise a tuple of 2 integers, indicates
        pruning with m_by_n sparsity, e.g., (2, 4): 2 zeros out of 4 consecutive
        elements. It check whether we can do pruning with m_by_n sparsity.
    """
    self._pruning_vars = pruning_vars
    self._pruning_schedule = pruning_schedule
    self._block_size = list(block_size)
    self._block_pooling_type = block_pooling_type
    self._sparsity_m_by_n = sparsity_m_by_n
    self._validate_block()

    # Training step
    self._step_fn = training_step_fn

    self._validate_block()

  def _validate_block(self):
    if self._block_size != [1, 1]:
      for weight, _, _ in self._pruning_vars:
        if weight.get_shape().ndims != 2:
          raise ValueError('Block Sparsity can only be used for layers which '
                           'have 2-dimensional weights.')

  def _update_mask(self, weights):
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
    sparsity = self._pruning_schedule(self._step_fn())[1]
    with tf.name_scope('pruning_ops'):
      abs_weights = tf.math.abs(weights)
      k = tf.dtypes.cast(
          tf.math.maximum(
              tf.math.round(
                  tf.dtypes.cast(tf.size(abs_weights), tf.float32) *
                  (1 - sparsity)),
              1),
          tf.int32)
      values, indices = tf.math.top_k(
          tf.reshape(abs_weights, [-1]), k=tf.size(abs_weights))

      # Grab the (k-1)th value as a threshold to build pruning mask.
      threshold_value = tf.gather(values, k - 1)
      threshold_pos = tf.gather(indices, k - 1)

      # Build mask for the weight element higher magnitude than threshold value.
      # A mask is added to make sure the threshold element be incorporated.
      # TODO(b/208967539): Update the logic to index oriented logic.
      new_mask = tf.math.logical_or(
          tf.math.greater_equal(abs_weights, threshold_value),
          tf.reshape(
              tf.one_hot(
                  threshold_pos,
                  depth=tf.size(abs_weights),
                  on_value=True,
                  off_value=False,
                  dtype=tf.bool), abs_weights.shape))

    # Updated mask is casted back to weight's data type in case of the type
    # mismatching due to keras mixed precision policy.
    return tf.dtypes.cast(threshold_value, weights.dtype), tf.dtypes.cast(
        new_mask, weights.dtype)

  def _update_mask_sparsity_m_by_n(self, weights, m_by_n=(2, 4)):
    """Updates the m by n sparsity mask for a given weight tensor.

    This function creates a mask for the given weight tensor so
    that n elements with the lowest absolute values in the block
    of m elements are set to be zero. We don't return any threshold.

    Args:
      weights: The weight tensor that needs to be masked.
      m_by_n: tuple of two integers, indicating m zeros out of n consecutive
        elements, default as 2 by 4 sparsity.

    Returns:
      new_mask: A numpy array of the same size and shape as weights containing
      0 or 1 to indicate which of the values in weights should be set to zero.
      It throws an error if the requested mask cannot be created.
    """
    prepared_weights = pruning_utils.weights_rearrange(weights)
    mask = pruning_utils.generate_m_by_n_mask(prepared_weights, m_by_n)
    new_mask = pruning_utils.m_by_n_sparsity_mask_prepare(mask, weights.shape)

    return new_mask

  def _maybe_update_block_mask(self, weights):
    """Performs block-granular masking of the weights.

    If sparsity_m_by_n is selected, then we return the relevant pruning mask,
    that nullify two out of four elements in the block.

    Block pruning occurs only if the block_height or block_width is > 1 and
    if the weight tensor, when squeezed, has ndims = 2. Otherwise, elementwise
    pruning occurs.
    Args:
      weights: The weight tensor that needs to be masked.

    Returns:
      new_threshold: The new value of the threshold based on weights, and
        sparsity at the current global_step. In case of sparsity m_by_n,
        the returned threshold is an arbitrary number.
      new_mask: A numpy array of the same size and shape as weights containing
        0 or 1 to indicate which of the values in weights falls below
        the threshold

    Raises:
      ValueError: if block pooling function is not AVG or MAX
    """
    if self._sparsity_m_by_n:
      mask = self._update_mask_sparsity_m_by_n(weights, self._sparsity_m_by_n)
      # We need to return some numbers for threshold.
      return 999.0, mask

    if self._block_size == [1, 1]:
      return self._update_mask(weights)

    # TODO(pulkitb): Check if squeeze operations should now be removed since
    # we are only accepting 2-D weights.

    squeezed_weights = tf.squeeze(weights)
    abs_weights = tf.math.abs(squeezed_weights)
    pooled_weights = pruning_utils.factorized_pool(
        abs_weights,
        window_shape=self._block_size,
        pooling_type=self._block_pooling_type,
        strides=self._block_size,
        padding='SAME')

    if pooled_weights.get_shape().ndims != 2:
      pooled_weights = tf.squeeze(pooled_weights)

    new_threshold, new_mask = self._update_mask(pooled_weights)

    updated_mask = pruning_utils.expand_tensor(new_mask, self._block_size)
    sliced_mask = tf.slice(
        updated_mask, [0, 0],
        [squeezed_weights.get_shape()[0],
         squeezed_weights.get_shape()[1]])
    return new_threshold, tf.reshape(sliced_mask, tf.shape(weights))

  def _weight_assign_objs(self):
    """Gather the assign objs for assigning weights<=weights*mask.

    The objs are ops for graph execution and tensors for eager
    execution.

    Returns:
      group of objs for weight assignment.
    """

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
        return tf_compat.assign(variable, reduced_value)

      update_objs = []
      for value, var in values_and_vars:
        update_objs.append(
            distribution.extended.update(var, update_var, args=(value,)))

      return tf.group(update_objs)

    assign_objs = []

    if tf.distribute.get_replica_context():
      values_and_vars = []
      for weight, mask, _ in self._pruning_vars:
        masked_weight = tf.dtypes.cast(
            tf.math.multiply(weight, mask), dtype=weight.dtype)
        values_and_vars.append((masked_weight, weight))
      if values_and_vars:
        assign_objs.append(tf.distribute.get_replica_context().merge_call(
            update_fn, args=(values_and_vars,)))
    else:
      for weight, mask, _ in self._pruning_vars:
        masked_weight = tf.dtypes.cast(
            tf.math.multiply(weight, mask), dtype=weight.dtype)
        assign_objs.append(tf_compat.assign(weight, masked_weight))

    return assign_objs

  def weight_mask_op(self):
    return tf.group(self._weight_assign_objs())

  def conditional_mask_update(self):
    """Returns an op to updates masks as per the pruning schedule."""

    def maybe_update_masks():
      if self._sparsity_m_by_n:
        # Update structured sparsity masks only at step 1
        return tf.math.equal(self._step_fn(), 1)
      else:
        return self._pruning_schedule(self._step_fn())[0]

    def no_update():
      return tf.no_op()

    def mask_update():
      """Updates mask without distribution strategy."""

      def update():
        assign_objs = []

        for weight, mask, threshold in self._pruning_vars:
          new_threshold, new_mask = self._maybe_update_block_mask(weight)
          assign_objs.append(tf_compat.assign(threshold, new_threshold))
          assign_objs.append(tf_compat.assign(mask, new_mask))

        return tf.group(assign_objs)

      return tf.cond(maybe_update_masks(), update, no_update)

    def mask_update_distributed(distribution):
      """Updates mask with distribution strategy."""

      def update(var, value):
        return tf_compat.assign(var, value)

      def update_distributed():
        """Gather distributed update objs.

        The objs are ops for graph execution and tensors for eager
        execution.
        """
        assign_objs = []

        for weight, mask, threshold in self._pruning_vars:
          new_threshold, new_mask = self._maybe_update_block_mask(weight)
          assign_objs.append(
              distribution.extended.update(mask, update, (new_mask,)))
          assign_objs.append(
              distribution.extended.update(threshold, update, (new_threshold,)))

        return tf.group(assign_objs)

      return tf.cond(maybe_update_masks(), update_distributed, no_update)

    if tf.distribute.get_replica_context():
      return tf.distribute.get_replica_context().merge_call(
          mask_update_distributed)
    else:
      return mask_update()

  def add_pruning_summaries(self):
    """Adds summaries of weight sparsities and thresholds."""
    # TODO(b/139939526): update to use public API.
    summary = tf.summary
    if not tf.executing_eagerly():
      summary = tf.compat.v1.summary
    summary.scalar('sparsity', self._pruning_schedule(self._step_fn())[1])
    for _, mask, threshold in self._pruning_vars:
      summary.scalar(mask.name + '/sparsity', 1.0 - tf.math.reduce_mean(mask))
      summary.scalar(threshold.name + '/threshold', threshold)
