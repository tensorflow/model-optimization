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
"""Default 8 bit Prune Preserve Quantization scheme which specifies how quantization should be applied."""

from tensorflow_model_optimization.python.core.quantization.keras.collaborative_optimizations.prune_preserve import (
    prune_preserve_quantize_registry)
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import (
    default_8bit_quantize_scheme,)


class Default8BitPrunePreserveQuantizeScheme(
    default_8bit_quantize_scheme.Default8BitQuantizeScheme):
  """Default 8 bit Prune Preserve Quantization Scheme."""

  def get_quantize_registry(self):
    return (prune_preserve_quantize_registry
            .Default8bitPrunePreserveQuantizeRegistry())

