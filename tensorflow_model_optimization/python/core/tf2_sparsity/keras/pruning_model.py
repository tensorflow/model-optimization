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
"""Keras callbacks for pruning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import g3
import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity_tf2 import pruning_wrapper


def _collect_prunable_layers(model):
  """Recursively collect the prunable layers in the model."""
  # TODO(Kaftan/xwinxu): Verify if this de-dupes layers,
  # and see if keras model.layers already flattens layers recursively or
  # not.
  prunable_layers = []
  for layer in model.layers:
    # A keras model may have other models as layers.
    if isinstance(layer, tf.keras.Model):
      prunable_layers += _collect_prunable_layers(layer)
    if isinstance(layer, pruning_wrapper.PrunableWrapper):
      prunable_layers.append(layer)

  return prunable_layers


class PruningModel(tf.keras.Model):
  """Keras model wrapped with pruning"""

  def __init__(self, model, pruner):
    super(tf.keras.Model, self).__init__()
    self.model = model
    self.pruner = pruner
    self.prunable_layers = []

  def call(self, *args, **kwargs):
    return self.model(**args, **kwargs)

  # TODO(kaftan/xwinxu): should this take a pruner in method signature?
  def prune(self, step):
    # Collect all the prunable layers in the model.
    if not self.prunable_layers:
      self.prunable_layers = _collect_prunable_layers(self.model)
    for prunable_layer in self.prunable_layers:
      self.pruner.update_masks(prunable_layer.pruning_vars, step)

  def train_step(self, data):
    super(PruningModel, self).train_step(data)
    self.prune(self.optimizer.iterations)

  # Todo: try out a pruning optimizer
