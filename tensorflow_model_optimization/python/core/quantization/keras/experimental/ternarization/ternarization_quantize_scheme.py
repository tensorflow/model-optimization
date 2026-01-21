# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow_model_optimization.python.core.quantization.keras.experimental.ternarization import ternarization_quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.experimental.ternarization import ternarization_quantize_registry


class TernarizationQuantizeScheme(quantize_scheme.QuantizeScheme):
  """Ternarization Scheme supported by TFLite."""

  def __init__(self, disable_per_axis=False):
    self._disable_per_axis = disable_per_axis

  def get_layout_transformer(self):
    return ternarization_quantize_layout_transform.TernarizationQuantizeLayoutTransform(
    )

  def get_quantize_registry(self):
    return (ternarization_quantize_registry.TernarizationQuantizeRegistry(
        disable_per_axis=self._disable_per_axis))
