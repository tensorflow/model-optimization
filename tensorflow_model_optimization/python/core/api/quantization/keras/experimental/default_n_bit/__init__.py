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
"""Module containing N-bit default quantization scheme."""
# pylint: disable=g-bad-import-order

# submodules
from tensorflow_model_optimization.python.core.api.quantization.keras.experimental.default_n_bit import default_n_bit_transforms

# The N-bit default quantization scheme classes.
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_quantize_scheme import DefaultNBitQuantizeScheme
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_quantize_layout_transform import DefaultNBitQuantizeLayoutTransform
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_quantize_registry import DefaultNBitQuantizeRegistry

# pylint: enable=g-bad-import-order
