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

from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from tensorflow_model_optimization.python.core.sparsity_tf2 import pruning_impl

keras = tf.keras
custom_object_scope = tf.keras.utils.custom_object_scope


# TODO serialization
# TODO for serialization: find some way to save dynamic
#  layer-specific logic in config? Might not be possible for an arbitrary
#  lambda?, but should be possible for 'common patterns' e.g. switching based
#  on layer type
class LowMagnitudePruningConfig(object):

  def __init__(
      self,
      pruning_schedule=pruning_sched.ConstantSparsity(0.5, 0),
      block_size=(1, 1),
      block_pooling_type='AVG'
  ):
    self._model = None
    self._variable_to_pruner_mapping = None
    self._pruner = pruning_impl.Pruner(pruning_schedule=pruning_schedule,
                                       block_size=block_size,
                                       block_pooling_type=block_pooling_type)

  def get_config(self):
    pass

  @classmethod
  def from_config(cls, config):
    pass

  def configure(self, model):
    self._model = model

  def _build_pruner_map(self):
    if self._model is None:
      raise ValueError('You may be using a PruningOptimizer without wrapping'
                       ' your model with a `PrunableModel`. You must configure'
                       ' it with a model to prune before you can'
                       ' look up a variable in a pruning configuration.'
                       ' `PrunableModel`s automatically configure'
                       ' when you compile them with a `PruningOptimizer`.')

    self._variable_to_pruner_mapping = dict()
    for var in self._model.trainable_weights:
      self._variable_to_pruner_mapping[var.ref()] = None

    def _process_layer(layer):
      for sub_layer in layer.layers:
        _process_layer(sub_layer)

      if isinstance(layer, prunable_layer.PrunableLayer):
        for var in layer.get_prunable_weights():
          self._variable_to_pruner_mapping[var.ref()] = self._pruner
      elif prune_registry.PruneRegistry.supports(layer):
        prune_registry.PruneRegistry.make_prunable(layer)
        for var in layer.get_prunable_weights():
          self._variable_to_pruner_mapping[var.ref()] = self._pruner

    _process_layer(self._model)

  def get_pruner(self, var):
    if not self._variable_to_pruner_mapping:
      self._build_pruner_map()

    var_ref = var.ref()
    if var_ref not in self._variable_to_pruner_mapping:
      raise ValueError('variable %s did not appear '
                       'in the configured model\'s trainable weights '
                       'the first time the pruning config tried to'
                       'look up a pruner for a variable.' % var.name)

    return self._variable_to_pruner_mapping[var_ref]
