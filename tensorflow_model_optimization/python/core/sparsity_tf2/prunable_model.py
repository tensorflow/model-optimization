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
import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity_tf2 import pruning_optimizer


class PrunableModel(tf.keras.Model):
  """Keras model that can be pruned by pruning optimizers"""

  def __init__(self, model, **kwargs):
    super(tf.keras.Model, self).__init__(**kwargs)
    self.model = model
    # TODO: Warn if the top-level model is a subclass model (not functional
    # or sequential) and does not extend PrunableLayer, because it
    # *may* include top-level weights/weights in nested
    # prunable layers that are prunable but aren't
    # registered.
    # also, clarify in prunablelayer whether get_prunable_weights
    # should include weights from nested layers or not

  def get_config(self):
    # Todo
    pass

  def from_config(cls, config, custom_objects=None):
    # todo
    pass

  def call(self, *args, **kwargs):
    return self.model(*args, **kwargs)

  def build(self, input_shape):
    self.model.build(input_shape)
    super(tf.keras.Model, self).build(input_shape)

  def compile(self, *args, **kwargs):
    super(tf.keras.Model, self).compile(*args, **kwargs)
    if isinstance(self.optimizer, pruning_optimizer.PruningOptimizer):
      self.optimizer.configure(self)
