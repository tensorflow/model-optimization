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
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry


class Default8BitQuantizeScheme(quantize_scheme.QuantizeScheme):

  def get_layout_transformer(self):
    return default_8bit_quantize_layout_transform.\
        Default8BitQuantizeLayoutTransform()

  def get_quantize_registry(self):
    return default_8bit_quantize_registry.Default8BitQuantizeRegistry()

