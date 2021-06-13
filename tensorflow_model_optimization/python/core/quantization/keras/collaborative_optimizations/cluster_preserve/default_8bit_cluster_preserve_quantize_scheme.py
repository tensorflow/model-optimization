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

"""Default 8 bit Cluster Preserve Quantization scheme."""

from tensorflow_model_optimization.python.core.quantization.keras.collaborative_optimizations.cluster_preserve import (
    cluster_preserve_quantize_registry,)
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_scheme


class Default8BitClusterPreserveQuantizeScheme(
    default_8bit_quantize_scheme.Default8BitQuantizeScheme):
  """Default 8 bit Cluster Preserve Quantization Scheme."""

  def __init__(self, preserve_sparsity=True):
    """Same as Default8BitQuantizeScheme but preserves clustering and sparsity.

    Args:
      preserve_sparsity: the flag to enable prune-cluster-preserving QAT.
    """
    self.preserve_sparsity = preserve_sparsity

  def get_quantize_registry(self):
    return (cluster_preserve_quantize_registry.
            Default8bitClusterPreserveQuantizeRegistry(self.preserve_sparsity))
