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
"""Keras callbacks for preserving sparsity in weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class PrunePreserve(tf.keras.callbacks.Callback):
  """Keras callback for PQAT, which preserves sparsity in weights.

  This callback must be used when optimizing the model with PQAT, it ensures
  sparsity of weights are preserved after the final backpropagation.

  Example:

  ```python
  quant_aware_model = tfmot.quantization.keras.quantize_apply(
      quant_aware_annotate_model,
      scheme=tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme())
  quant_aware_model.fit(x, y,
      callbacks=[tfmot.experimental.combine.PrunePreserve()])
  ```
  """

  def __init__(self):
    super(PrunePreserve, self).__init__()

  def on_epoch_end(self, batch, logs=None):
    # At the end of every epoch, reapply sparsity masks. This ensures that when
    # the model is saved after completion, the sparsity of weights are preserved.
    for layer in self.model.layers:
      if hasattr(layer, 'prune_preserve_vars'):
        masked_weights = tf.multiply(
            layer.prune_preserve_vars['weights'],
            layer.prune_preserve_vars['sparsity_mask'])
        layer.prune_preserve_vars['weights'].assign(masked_weights)
