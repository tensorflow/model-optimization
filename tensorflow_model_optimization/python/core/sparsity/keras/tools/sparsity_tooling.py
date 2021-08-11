# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities to prune without training.

Quickly produces pruned models, with no concern for accuracy. Useful to
evaluate the performance benefits of given pruning parameters, without
time-consuming retraining.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.keras import metrics
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

keras = tf.keras


class StepIndependentConstantSparsity(pruning_schedule.PruningSchedule):
  """Pruning schedule with constant sparsity, applied at any step."""

  def __init__(self, target_sparsity):
    """Initializes a Pruning schedule with constant sparsity.

    Sparsity is applied at every step.

    Args:
      target_sparsity: Target sparsity as float, in [0, 1] interval.
    """
    self.target_sparsity = target_sparsity

  def __call__(self, step):
    return (True, tf.constant(self.target_sparsity, dtype=tf.float32))

  def get_config(self):
    return {
        'class_name': self.__class__.__name__,
        'config': {
            'target_sparsity': self.target_sparsity,
        }
    }


def _apply_pruning(prunable_object):
  """Calculates the masks and updates weights of layers of a wrapped model."""
  assert tf.executing_eagerly()
  for layer in pruning_wrapper.collect_prunable_layers(prunable_object):
    layer.pruning_obj.conditional_mask_update()  # Create mask
    layer.pruning_obj.weight_mask_op()  # weight = weight * mask


@metrics.MonitorBoolGauge('prune_for_benchmark_usage')
def prune_for_benchmark(keras_model,
                        target_sparsity,
                        block_size=(1, 1)):
  """Prunes a tf.keras model in a single step, without re-training.

  This function is intented to quickly apply sparsity to a model, without
  consideration for accuracy.

  Args:
    keras_model: A `tf.keras.Model` instance.
    target_sparsity: Target sparsity as float, in [0, 1] interval.
    block_size: The dimensions (height, weight) for the block sparse
      pattern in rank-2 weight tensors.
  Returns:
    A pruned model, modified with pruning wrappers.
  """

  pruning_params = {
      'pruning_schedule': StepIndependentConstantSparsity(target_sparsity),
      'block_size': block_size,
  }

  prunable_object = prune.prune_low_magnitude(keras_model, **pruning_params)
  _apply_pruning(prunable_object)

  return prunable_object
