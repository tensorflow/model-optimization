# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Quantization scheme which specifies how quantization should be applied."""

from tensorflow_model_optimization.python.core.quantization.keras import quantize_scheme
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import default_n_bit_quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import default_n_bit_quantize_registry


class DefaultNBitQuantizeScheme(quantize_scheme.QuantizeScheme):
  """Default N-Bit Scheme supported by TFLite."""

  def __init__(self, disable_per_axis=False,
               num_bits_weight=8, num_bits_activation=8):
    self._disable_per_axis = disable_per_axis
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

  def get_layout_transformer(self):
    return default_n_bit_quantize_layout_transform.DefaultNBitQuantizeLayoutTransform(
        num_bits_weight=self._num_bits_weight,
        num_bits_activation=self._num_bits_activation)

  def get_quantize_registry(self):
    return (
        default_n_bit_quantize_registry.DefaultNBitQuantizeRegistry(
            disable_per_axis=self._disable_per_axis,
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))
