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
"""Default N-Bit QuantizeConfigs."""

from typing import Any, Dict
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
from tensorflow_model_optimization.python.core.quantization.keras import quantizers


class DefaultNBitOutputQuantizeConfig(quantize_config.QuantizeConfig):
  """QuantizeConfig which only quantizes the output from a layer."""

  def __init__(self, num_bits_weight: int = 8, num_bits_activation: int = 8):
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    return [quantizers.MovingAverageQuantizer(
        num_bits=self._num_bits_activation, per_axis=False,
        symmetric=False, narrow_range=False)]  # activation/output

  def get_config(self) -> Dict[str, Any]:
    return {
        'num_bits_weight': self._num_bits_weight,
        'num_bits_activation': self._num_bits_activation,
    }


class NoOpQuantizeConfig(quantize_config.QuantizeConfig):
  """QuantizeConfig which does not quantize any part of the layer."""

  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}
