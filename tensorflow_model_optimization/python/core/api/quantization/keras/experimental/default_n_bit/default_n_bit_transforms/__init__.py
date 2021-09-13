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
"""Module containing N-bit default transforms."""

# The 8bit default transform classes.
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import ConcatTransform
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import ConcatTransform3Inputs
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import ConcatTransform4Inputs
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import ConcatTransform5Inputs
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import ConcatTransform6Inputs
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import Conv2DBatchNormActivationQuantize
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import Conv2DBatchNormQuantize
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import Conv2DBatchNormReLUQuantize
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import Conv2DReshapeBatchNormActivationQuantize
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import Conv2DReshapeBatchNormQuantize
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import Conv2DReshapeBatchNormReLUQuantize
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import InputLayerQuantize
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import LayerReluActivationQuantize
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import LayerReLUQuantize
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import SeparableConv1DQuantize
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_transforms import SeparableConvQuantize
