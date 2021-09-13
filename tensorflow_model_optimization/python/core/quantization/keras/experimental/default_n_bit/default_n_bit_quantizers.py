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
"""Quantizers specific to default 8-bit behavior."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantizers


class DefaultNBitConvWeightsQuantizer(quantizers.LastValueQuantizer):
  """Quantizer for handling weights in Conv2D/DepthwiseConv2D layers."""

  def __init__(self, num_bits_weight: int = 8, num_bits_activation: int = 8):
    """Construct LastValueQuantizer with params specific for TFLite Convs."""

    super(DefaultNBitConvWeightsQuantizer, self).__init__(
        num_bits=num_bits_weight,
        per_axis=True,
        symmetric=True,
        narrow_range=True)  # weight
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

  def build(self, tensor_shape, name, layer):
    min_weight = layer.add_weight(
        name + '_min',
        shape=(tensor_shape[-1],),
        initializer=tf.keras.initializers.Constant(-6.0),
        trainable=False)
    max_weight = layer.add_weight(
        name + '_max',
        shape=(tensor_shape[-1],),
        initializer=tf.keras.initializers.Constant(6.0),
        trainable=False)

    return {'min_var': min_weight, 'max_var': max_weight}


class DefaultNBitConvTransposeWeightsQuantizer(quantizers.LastValueQuantizer):
  """Quantizer for handling weights in Conv2DTranspose layers."""

  def __init__(self, num_bits_weight: int = 8, num_bits_activation: int = 8):
    """Construct LastValueQuantizer with params specific for TFLite Conv2DTranpose."""
    super(DefaultNBitConvTransposeWeightsQuantizer, self).__init__(
        num_bits=num_bits_weight,
        per_axis=False,
        symmetric=True,
        narrow_range=True)  # weight
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

  def __call__(self, inputs, training, weights, **kwargs):
    outputs = tf.transpose(inputs, (0, 1, 3, 2))
    outputs = super(DefaultNBitConvTransposeWeightsQuantizer,
                    self).__call__(outputs, training, weights, **kwargs)
    outputs = tf.transpose(outputs, (0, 1, 3, 2))
    return outputs
