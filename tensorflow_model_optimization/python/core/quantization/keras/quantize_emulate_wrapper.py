# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Quantization Emulation Wrapper.

   The QuantizeEmulateWrapper keras layer wrapper simulates inference time
   quantization for a particular layer of the neural network during training.
   This adds quantization loss errors during the training process and allows
   the model to learn parameters to be more robust to them during training.
"""

from tensorflow.python.framework import ops
from tensorflow.python.keras.layers.wrappers import Wrapper
from tensorflow_model_optimization.python.core.quantization.keras import quant_ops
from tensorflow_model_optimization.python.core.quantization.keras.quantize_emulatable_layer import QuantizeEmulatableLayer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_emulate_registry import QuantizeEmulateRegistry


class QuantizeEmulateWrapper(Wrapper):
  """Quantizes the weights/output of the keras layer it wraps."""

  def __init__(self,
               layer,
               num_bits,
               narrow_range=True,
               symmetric=True,
               **kwargs):
    """Create a quantize emulate wrapper for a keras layer.

    Args:
      layer: The keras layer to be quantized.
      num_bits: Number of bits for quantization
      narrow_range: Whether to use the narrow quantization range [1; 2^num_bits
        - 1] or wide range [0; 2^num_bits - 1].
      symmetric: If true, use symmetric quantization limits instead of training
        the minimum and maximum of each quantization range separately.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    if isinstance(layer, QuantizeEmulatableLayer):
      # Custom layer in client code which supports quantization.
      super(QuantizeEmulateWrapper, self).__init__(layer, **kwargs)
    elif QuantizeEmulateRegistry.supports(layer):
      # Built-in keras layers which support quantize emulation.
      super(QuantizeEmulateWrapper, self).__init__(
          QuantizeEmulateRegistry.make_quantizable(layer), **kwargs)
    else:
      raise ValueError("Unsupported Layer " + layer.__class__)

    self.num_bits = num_bits
    self.symmetric = symmetric
    self.narrow_range = narrow_range

    self.unquantized_kernels = []
    self.quantized_kernels = []

  def build(self, input_shape):
    self.layer.build(input_shape)

    super(QuantizeEmulateWrapper, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(self.layer.input_shape)

  def call(self, inputs, **kwargs):
    for unquantized_kernel in self.layer.get_quantizable_weights():
      # Quantize the layer's weights and assign the resulting tensor to the
      # layer. This ensures the results of the forward pass use the quantized
      # tensor value. However, the internal _trainable_weights is not modified
      # since the unquantized weights need to be updated.
      quantized_kernel = quant_ops.LastValueQuantize(
          unquantized_kernel,
          init_min=-6.0,
          init_max=6.0,
          is_training=True,
          num_bits=self.num_bits,
          symmetric=self.symmetric,
          narrow_range=self.narrow_range,
          vars_collection=ops.GraphKeys.GLOBAL_VARIABLES,
          name_prefix=self.layer.name)

      self.unquantized_kernels.append(unquantized_kernel)
      self.quantized_kernels.append(quantized_kernel)

    self.layer.set_quantizable_weights(self.quantized_kernels)

    outputs = self.layer.call(inputs, **kwargs)

    if self.layer.activation is None:
      return outputs

    outputs = quant_ops.MovingAvgQuantize(
        outputs,
        init_min=-6.0,
        init_max=6.0,
        ema_decay=0.999,
        is_training=True,
        num_bits=self.num_bits,
        symmetric=self.symmetric,
        narrow_range=self.narrow_range,
        vars_collection=ops.GraphKeys.GLOBAL_VARIABLES,
        name_prefix=self.layer.name)

    return outputs
