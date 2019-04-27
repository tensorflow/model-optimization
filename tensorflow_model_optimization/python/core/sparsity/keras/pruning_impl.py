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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.summary import summary as summary_ops_v1
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils


class Pruning(object):
  """Implementation of magnitude-based weight pruning."""

  def __init__(self, training_step_fn, pruning_vars, pruning_schedule,
               block_size, block_pooling_type):
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
    """
    self._pruning_vars = pruning_vars
    self._pruning_schedule = pruning_schedule
    self._block_size = list(block_size)
    self._block_pooling_type = block_pooling_type
    self._validate_block()

    # List of tensorflow assignments ops for new masks and thresholds
    self._assign_ops = []

    # Training step
    self._step_fn = training_step_fn

    # List of tensorflow assignment ops for the weights
    self._weight_assign_ops = []

    self._validate_block()

  def _validate_block(self):
    if self._block_size != [1, 1]:
      for weight, _, _ in self._pruning_vars:
        if weight.get_shape().ndims != 2:
          raise ValueError('Block Sparsity can only be used for layers which '
                           'have 2-dimensional weights.')

  def get_weight_sparsity(self):
    return [math_ops.reduce_mean(weight) for weight, _, _ in self._pruning_vars]

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
    with ops.name_scope('pruning_ops'):
      abs_weights = math_ops.abs(weights)
      k = math_ops.cast(
          math_ops.round(
              math_ops.cast(array_ops.size(abs_weights), dtypes.float32) *
              (1 - sparsity)), dtypes.int32)
      # Sort the entire array
      values, _ = nn_ops.top_k(
          array_ops.reshape(abs_weights, [-1]), k=array_ops.size(abs_weights))
      # Grab the (k-1)th value

      current_threshold = array_ops.gather(values, k - 1)
      new_mask = math_ops.cast(
          math_ops.greater_equal(abs_weights, current_threshold),
          dtypes.float32)
    return current_threshold, new_mask

  def _maybe_update_block_mask(self, weights):
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
      return self._update_mask(weights)

    # TODO(pulkitb): Check if squeeze operations should now be removed since
    # we are only accepting 2-D weights.

    squeezed_weights = array_ops.squeeze(weights)
    abs_weights = math_ops.abs(squeezed_weights)
    pooled_weights = pruning_utils.factorized_pool(
        abs_weights,
        window_shape=self._block_size,
        pooling_type=self._block_pooling_type,
        strides=self._block_size,
        padding='SAME')

    if pooled_weights.get_shape().ndims != 2:
      pooled_weights = array_ops.squeeze(pooled_weights)

    new_threshold, new_mask = self._update_mask(pooled_weights)

    updated_mask = pruning_utils.expand_tensor(new_mask, self._block_size)
    sliced_mask = array_ops.slice(
        updated_mask, [0, 0],
        [squeezed_weights.get_shape()[0],
         squeezed_weights.get_shape()[1]])
    return new_threshold, array_ops.reshape(sliced_mask,
                                            array_ops.shape(weights))

  def _get_assign_ops(self):
    """Gather the assign ops for assigning updated masks and threshold."""
    # Make sure the assignment ops have not already been added to the list
    if self._assign_ops:
      raise ValueError(
          'Assign op list not empty. _get_assign_ops() called twice?')

    for weight, mask, threshold in self._pruning_vars:
      is_partitioned = isinstance(weight, variables.PartitionedVariable)
      weight_as_tensor = weight
      if is_partitioned:
        weight_as_tensor = weight.as_tensor()

      new_threshold, new_mask = self._maybe_update_block_mask(weight_as_tensor)
      self._assign_ops.append(
          pruning_utils.variable_assign(threshold, new_threshold))

      self._assign_ops.append(
          pruning_utils.partitioned_variable_assign(mask, new_mask)
          if is_partitioned else pruning_utils.variable_assign(mask, new_mask))

  def _get_weight_assign_ops(self):
    """Gather the assign ops for assigning weights<=weights*mask."""
    if self._weight_assign_ops:
      raise ValueError(
          'Assign op list not empty. _get_weight_assign_ops() called twice?')

    for weight, mask, _ in self._pruning_vars:
      is_partitioned = isinstance(weight, variables.PartitionedVariable)
      masked_weight = math_ops.multiply(weight, mask)
      self._weight_assign_ops.append(
          pruning_utils.partitioned_variable_assign(weight, masked_weight)
          if is_partitioned else pruning_utils
          .variable_assign(weight, masked_weight))

  def weight_mask_op(self):
    if tf.executing_eagerly() or not self._weight_assign_ops:
      self._weight_assign_ops = []
      self._get_weight_assign_ops()

    with ops.control_dependencies(self._weight_assign_ops):
      return control_flow_ops.no_op('mask_weights')

  def mask_update_op(self):
    self._assign_ops = []
    self._get_assign_ops()

    with ops.control_dependencies(self._assign_ops):
      return control_flow_ops.no_op('mask_update')

  def conditional_mask_update(self):
    """Returns an op to updates masks as per the pruning schedule."""

    def maybe_update_masks():
      return self._pruning_schedule(self._step_fn())[0]

    def mask_update_op():
      return self.mask_update_op()

    def no_op():
      return control_flow_ops.no_op()

    return control_flow_ops.cond(maybe_update_masks(), mask_update_op, no_op)

  def add_pruning_summaries(self):
    """Adds summaries of weight sparsities and thresholds."""
    summary = summary_ops_v1
    if tf.executing_eagerly():
      summary = summary_ops_v2
    summary.scalar('sparsity', self._pruning_schedule(self._step_fn())[1])
    for _, mask, threshold in self._pruning_vars:
      summary.scalar(mask.name + '/sparsity', 1.0 - math_ops.reduce_mean(mask))
      summary.scalar(threshold.name + '/threshold', threshold)
