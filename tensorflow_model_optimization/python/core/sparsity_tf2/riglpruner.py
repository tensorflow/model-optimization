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
"""Helper functions to add support for iterative magnitude/gradient pruning as seen in the RiGL experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import logging

from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.summary import summary as summary_ops_v1
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils
from tensorflow_model_optimization.python.core.sparsity_tf2 import schedule as update_schedule
from tensorflow_model_optimization.python.core.sparsity_tf2 import pruner
from tensorflow_model_optimization.python.core.sparsity_tf2 import sparse_utils


class RiGLPruner(pruner.Pruner):
  """
  Implementation of the RiGL dynamic sparse training algorithm.
  """

  def __init__(self,
      update_schedule=update_schedule.ConstantSchedule(0.5, 0),
      sparsity=0.5,
      block_size=(1,1),
      block_pooling_type='AVG',
      stateless=True,
      seed=0,
      noise_std=0,
      reinit=False
    ):
    """The logic for magnitude-based RiGL trained weight tensors as presented
    in https://proceedings.icml.cc/static/paper_files/icml/2020/287-Paper.pdf.
    At each iteration, according to the update_schedule, connections are grown
    based on gradient information and dropped based on weight magnitude.

    Args:
      update_schedule: A `PruningSchedule` object that controls pruning rate
        throughout training.
      sparsity: layer specific sparsity at which the dynamic sparse training method uses
      block_size: The dimensions (height, weight) for the block sparse pattern
        in rank-2 weight tensors.
      block_pooling_type: (optional) The function to use to pool weights in the
        block. Must be 'AVG' or 'MAX'.
      stateless: whether or not being run on TPU/multi-GPU workers.
      seed: assigned by PruningConfig (base added) for multiworker consistency in random processes.
      reinit: boolean for whether to reinitialize a connection that was drop and regrown to 
        its original value, or to set it to 0
    """
    super(RiGLPruner, self).__init__(
        update_schedule, block_size, block_pooling_type)
    self.target_sparsity = sparsity
    self.update_schedule = update_schedule
    self._block_size = block_size
    self.block_pooling_type = block_pooling_type
    self._stateless = stateless
    self._seed = seed
    self._noise_std = noise_std
    self._reinit_when_same = reinit
  
  def create_slots(self, optimizer, var):
    base_dtype = var.dtype
    deterministic_initializer = sparse_utils.PermuteOnes(self.target_sparsity)
    optimizer.add_slot(var, 'mask', initializer=deterministic_initializer)

  def _validate_block(self, var):
    if self._block_size != [1, 1]:
      if var.get_shape().ndims != 2:
        raise ValueError('Block Sparsity can only be used for layers which '
                          'have 2-dimensional weights.')

  def _random_normal(self, step, *args, **kwargs):
    """Gaussian noise distribution"""
    if self._stateless:
      offset_seed = self._seed_offset + kwargs.get('seed', 0)
      kwargs['seed'] = tf.cast(
        tf.stack([offset_seed, step], tf.int32)
      )
      return tf.random.stateless_normal(*args, **kwargs)
    return tf.random.normal(*args, **kwargs)

  def _get_grow_grads(self, optimizer, step, mask, grad):
    """Grow connections baased on gradient information.
    """
    grad_scores = tf.math.abs(grad)
    return grad_scores

  def _get_drop_weights(self, optimizer, step, mask, weight, noise_std=0):
    """
    Drop connections based on weight magnitude.
    """
    masked_weight = mask * weight
    weight_scores = tf.math.abs(masked_weight)
    # add noise to break ties, although the possibility is very low
    if noise_std != 0:
      noise = self._random_normal(
        weight_scores.shape, stddev=noise_std, dtype=weight_scores.dtype,
        seed=self._seed)
      weight_scores = weight_scores + noise
    return weight_scores

  def _reset_momentum(self, optimizer, weight, new_connections):
    """Zeros out optimizer slots whose connections have been recovered.
    This is done so that the aggregated values are zeroed out.
    """
    for slot_name in optimizer._optimizer.get_slot_names():
      # reset aggregated momentum variables to 0
      opt_var = optimizer._optimizer.get_slot(weight, slot_name)
      new_values = tf.where(new_connections,
                            tf.zeros_like(opt_var), opt_var)
      opt_var.assign(new_values)

  def _generic_top_k(self, scores, mask, n_new, n_total):
     # sort the entire array so that it can be constant for TPUs
    _, sorted_idx = tf.math.top_k(
      tf.reshape(scores, (-1), k=n_total)
    )
    expanded_sorted_idx = tf.expand_dims(sorted_idx, 1)
    new_values = tf.where(
      tf.range(n_total) < n_new,
      tf.ones_like(sorted_idx, dtype=mask.dtype),
      tf.zeros_like(sorted_idx, dtype=mask.dtype)
    )
    updated_mask = tf.scatter_nd(expanded_sorted_idx, new_values, new_values.shape)

    return updated_mask

  def _get_new_connections(self, reinit_when_same, grown_mask_reshaped, mask):
    """When dropped and regrown connections are the same there are two options:
      1. keep original value
      2. set it to 0
      3. (not implemented, but an option) use gradient direction
    """
    if reinit_when_same:
      new_connections = tf.math.equal(grown_mask_reshaped, 1)
    else:
      new_connections = tf.math.logical_and(
        tf.math.equal(grown_mask_reshaped, 1), tf.math.equal(mask, 0)
      )
    return new_connections

  def _update_mask(self, step, update_fraction, mask, weight, grad):
    """Called by _maybe_update_block_mask.
    Updates mask based on weight and grad information.

    Args:
      step: current iteration from optimizer
      update_fraction: the fraction of existing connections to update
      mask: the mask to update
      weight: trainable variable representing weights
      grad: gradients obtained from the optimizer
    """
    # compute the top k magnitudes then update the current mask
    drop_scores = self._get_drop_weights(optimizer, step, mask, weight, noise_std=self._noise_std)
    # need access to exactly which entries are growing to zero out optimizer slot
    grow_scores = self._get_grow_grads(optimizer, step, mask, grad)
    n_total = tf.size(drop_scores)
    n_ones = tf.cast(tf.reduce_sum(mask), dtype=tf.int32)
    n_prune = tf.cast(
      tf.cast(n_ones, dtype=tf.float32) * update_fraction, tf.int32
    )
    n_keep = n_ones - n_prune

    dropped_mask = self._generic_top_k(drop_scores, mask, n_keep, n_total)

    if grow_scores is not None:
      # flatten the scores
      grow_scores = tf.reshape(grow_scores, (-1))
      # set enabled connections (ones) to min(scores) - 1, i.e. they have the lowest scores
      grow_scores_lifted = tf.where(
        tf.math.equal(dropped_mask, 1),
        tf.ones_like(dropped_mask) * (tf.reduce_min(grow_scores) - 1), grow_scores
      )
      grown_mask = self._generic_top_k(grow_scores_lifted, mask, n_prune, n_total)
      # ensure that masks are disjoint
      tf.debugging.Assert(
        tf.math.equal(tf.reduce_sum(dropped_mask * grown_mask, 0.), [dropped_mask, grown_mask])
      )

      grown_mask_reshaped = tf.reshape(grown_mask, mask.shape)
      # set the values of the new connections
      grow_tensor = tf.zeros_like(weight, dtype=weight.dtype)
      new_connections = self._get_new_connections(self._reinit_when_same, grown_mask_reshaped, mask)

      new_weights = tf.where(new_connections, grow_tensor, weight)
      # update weights
      weight.assign(new_weights)
      reset_momentum = True
      mask_combined = tf.reshape(dropped_mask + grown_mask, mask.shape)
    else:
      mask_combined = tf.reshape(dropped_mask, mask.shape)
      reset_momentum = False
      new_connections = None

    return reset_momentum, mask_combined, new_connections

  
  def _maybe_update_block_mask(self, step, update_fraction, mask, weights, grads):
    """Performs block-granular masking of the weights.

    Block pruning occurs only if the block_height or block_width is > 1 and
    if the weight tensor, when squeezed, has ndims = 2. Otherwise, elementwise
    pruning occurs.
    Args:
      step: the step tensor at which to update the mask.
      weights: The weight tensor that needs to be masked.
      update_fraction: the current fraction of connections 
        of which to potentially update (i.e. drop / grow)
      mask: the current mask stored in the optimizer slot
      grads: the gradient with respect to the current weights

    Returns:
      reset_momentum: A boolean indicating whether or not momentum slot of 
        optimizer needs to be reset (i.e. connections have been grown)
      new_mask: A numpy array of the same size and shape as weights containing
        0 or 1 to indicate which of the values in weights were pruned/dropped
      new_connections: A numpy array of the same size and shape as weights
        containing 0 or 1 to indicate which of the values in weights
        were grown

    Raises:
      ValueError: if block pooling function is not AVG or MAX
    """
    if self._block_size == [1, 1]:
      return self._update_mask(step, update_fraction, mask, weights, grads)

    abs_weights = tf.math.abs(weights)
    pooled_weights = pruning_utils.factorized_pool(
        abs_weights,
        window_shape=self._block_size,
        pooling_type=self._block_pooling_type,
        strides=self._block_size,
        padding='SAME')

    if pooled_weights.get_shape().ndims != 2:
      pooled_weights = tf.squeeze(pooled_weights)

    # TODO(xwinxu): confirm how the mask works for pooled_weights
    reset_momentum, new_mask, new_connections = self._update_mask(step, update_fraction, mask, pooled_weights)

    updated_mask = pruning_utils.expand_tensor(new_mask, self._block_size)
    sliced_mask = tf.slice(
        updated_mask, [0, 0],
        [weights.get_shape()[0],
         weights.get_shape()[1]])
    return reset_momentum, tf.reshape(sliced_mask, tf.shape(weights)), new_connections


  def update_masks(self, update_vars, step):
    """Updates masks as per the update schedule.

    Args: 
      update_vars: A list of (update_fraction, mask, weight, gradient) tuples
      step: the current iteration of the optimizer

    Returns:
      reset_momentum: boolean indicating whether or not connections 
        were grown and thus if momentum needs to be nullified
      new_connections: if reset_momentum is True, the a binary mask for
        where new weights were re-introduced into the mask
    """
    self._validate_block(var)
    should_update, update_fraction = self.update_schedule(step)
    new_connections = None
    reset_momentum = False
    if should_update:
      for mask, weight, grad in update_vars:
        reset_momentum, new_mask, new_connections = self._maybe_update_block_mask(step, update_fraction, mask, weight, grad)
        mask.assign(new_mask)

    return reset_momentum, new_connections
  
  def _apply_mask(self, weight, mask):
    """Directly masks the weights (updating the weight variables)."""

    # TODO(Kaftan/xwinxu): figure out if this is totally unneeded now
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
  

  def preprocess_weights(self, optimizer, var, grad):
    """Apply gradient update before the first weight update, 
    so that you don't save at start of current round specified.
    """
    # gradient is unused for lottery ticket pruning, but may be masked for others
    return grad

  def postprocess_weights(self, optimizer, var, grad):
    """Update the optimizer components after the weights have been updated by the optimizer.
    """
    mask = optimizer.get_slot(var, 'mask')
    # calculate new connections and assign mask
    reset_momentum, new_connections = self.update_masks([(mask, var, grad)], step=optimizer.iterations)
    # assign weights based on new sparse binary mask
    self._apply_mask(var, mask)
    # ensure there is not momentum values for new connections
    if reset_momentum:
      self._reset_momentum(optimizer, weight, new_connections)
    