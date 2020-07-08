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
"""Helper functions to add support for iterative magnitude pruning as seen in the Lottery Ticket Hypothesis experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import logging

from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.summary import summary as summary_ops_v1
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from tensorflow_model_optimization.python.core.sparsity_tf2 import pruner


class LTHPruner(pruner.Pruner):
  """
  Implementation of the lottery ticket iterative magnitude pruning algorithm.
  """

  def __init__(self,
      pruning_schedule=pruning_sched.ConstantSparsity(0.5, 0),
      save_iteration=None,
      block_size=(1,1),
      block_pooling_type='AVG',
  ):
    """The logic for magnitude-based pruning weight tensors.

    Args:
      pruning_schedule: A `PruningSchedule` object that controls pruning rate
        throughout training.
      save_iteration: A integer representing the weights for relloading after checkpointing in LTH experiments.
      block_size: The dimensions (height, weight) for the block sparse pattern
        in rank-2 weight tensors.
      block_pooling_type: (optional) The function to use to pool weights in the
        block. Must be 'AVG' or 'MAX'.
    """
    super().__init__(pruning_schedule, block_size, block_pooling_type)
    self._reload_schedule = pruning_schedule
    self._save_step = save_iteration if save_iteration else 0

    if not isinstance(self._reload_schedule, pruning_sched.PruningSchedule) and self._save_step > 0:
      logging.warning(f"We could not statically verify if the lottery ticket initializations will be" + 
      " saved before the first pruning. The weights will be saved at iteration {self._save_step} but the pruning" +
      " schedule is not an analyzable PruningSchedule object.")
    if (isinstance(self._reload_schedule, pruning_sched.PruningSchedule) 
      and self._save_step > self._reload_schedule.begin_step):
      raise ValueError("Reloading should not occur before initializations are saved.")
  
  def create_slots(self, optimizer, var):
    base_dtype = var.dtype
    optimizer.add_slot(var, 'mask', initializer='ones')
    optimizer.add_slot(var, 'threshold', initializer=tf.zeros(shape=(), dtype=base_dtype))
    optimizer.add_slot(var, 'original_initialization', initializer=var.read_value())

  def _maybe_save_weights(self, optimizer, var):
    """Save the weights at the desired iteration as the original initialization.
    """
    if self._save_step == optimizer.iterations:
      optimizer.get_slot(var, 'original_initialization').assign(var)
      
  def _maybe_reload_weights(self, optimizer, var, mask):
    """
    Reload weights according to the pruning schedule, unless specified otherwise by `reload_schedule`.
    """
    should_prune = self._reload_schedule._should_prune_in_step(optimizer.iterations,
                    self._reload_schedule.begin_step, self._reload_schedule.end_step, self._reload_schedule.frequency)
    if should_prune:
      reload_weights = tf.math.multiply(optimizer.get_slot(var, 'original_initialization'), mask)
      var.assign(reload_weights)
  
  def preprocess_weights(self, optimizer, var, grad):
    """Apply gradient update before the first weight update, so that you don't save at start of current round specified.
    """
    # gradient is unused for lottery ticket pruning, but may be masked for others
    self._maybe_save_weights(optimizer, var)
    return grad

  def postprocess_weights(self, optimizer, var, grad):
    """Update the optimizer components after the weights have been updated by the optimizer.
    """
    mask = optimizer.get_slot(var, 'mask')
    threshold = optimizer.get_slot(var, 'threshold')
    self.update_masks([(var, mask, threshold)], step=optimizer.iterations)
    self._maybe_reload_weights(optimizer, var, mask)
    self._apply_mask(var, mask)
