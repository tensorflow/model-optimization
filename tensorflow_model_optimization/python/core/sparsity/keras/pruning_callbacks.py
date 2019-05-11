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

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.ops import math_ops
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper


class UpdatePruningStep(callbacks.Callback):
  """Keras callback which updates pruning wrappers with the optimizer step.

  This callback must be used when training a model which needs to be pruned. Not
  doing so will throw an error.

  Example:

  ```python
  model.fit(x, y,
      callbacks=[UpdatePruningStep()])
  ```
  """

  def on_train_begin(self, logs=None):
    self.step = K.get_value(self.model.optimizer.iterations)

  def on_train_batch_begin(self, batch, logs=None):
    tuples = []
    for layer in self.model.layers:
      # TODO(pulkitb): There's a possibility that the layer is a wrapper which
      # internally contains Prune. This should account for that. Else Prune
      # wrappers will throw errors.
      if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        # Assign iteration count from the optimizer to the layer pruning_step.
        tuples.append((layer.pruning_step, self.step))

    K.batch_set_value(tuples)
    self.step = self.step + 1

  def on_epoch_end(self, batch, logs=None):
    # At the end of every epoch, remask the weights. This ensures that when
    # the model is saved after completion, the weights represent mask*weights.
    layers = self.model.layers
    weight_mask_ops = []

    for layer in layers:
      if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        if tf.executing_eagerly():
          layer.pruning_obj.weight_mask_op()
        else:
          weight_mask_ops.append(layer.pruning_obj.weight_mask_op())

    K.batch_get_value(weight_mask_ops)


class PruningSummaries(callbacks.TensorBoard):
  """A Keras callback for adding pruning summaries to tensorboard.

  Logs the sparsity(%) and threshold at a given iteration step.
  """

  def __init__(self, log_dir, update_freq='epoch', **kwargs):
    super(PruningSummaries, self).__init__(
        log_dir=log_dir, update_freq=update_freq, **kwargs)

  def on_epoch_end(self, batch, logs=None):
    super(PruningSummaries, self).on_epoch_end(batch, logs)

    pruning_logs = {}
    params = []
    layers = self.model.layers
    for layer in layers:
      if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        for _, mask, threshold in layer.pruning_vars:
          params.append(mask)
          params.append(threshold)
    params.append(self.model.optimizer.iterations)

    values = K.batch_get_value(params)
    iteration = values[-1]
    del values[-1]
    del params[-1]

    param_value_pairs = list(zip(params, values))

    for mask, mask_value in param_value_pairs[::2]:
      pruning_logs.update({
          mask.name + '/sparsity': 1 - np.mean(mask_value)
      })

    for threshold, threshold_value in param_value_pairs[1::2]:
      pruning_logs.update({threshold.name + '/threshold': threshold_value})

    self._log_metrics(pruning_logs, '', iteration)
