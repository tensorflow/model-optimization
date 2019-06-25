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
"""Activation layer which applies emulates quantization during training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers import Layer

from tensorflow_model_optimization.python.core.quantization.keras import quant_ops


class QuantizeAwareActivation(Layer):
  """Activation layer for quantization aware training.

  The goal of this layer is to apply quantize operations during training such
  that the training network mimics quantization loss experienced in activations
  during inference.

  It introduces quantization loss before and after activations as required to
  mimic inference loss. The layer has built-in knowledge of how quantized
  activations are laid out during inference to emulate exact behavior.

  For example, ReLU activations are typically fused into their parent layer
  such as Conv/Dense. Hence, loss is introduced only after the activation has
  been applied. For Softmax on the other hand quantization loss is experienced
  both before and after the activation.

  Input shape:
    Arbitrary.

  Output shape:
    Same shape as input.
  """

  _PRE_ACTIVATION_TYPES = {'softmax'}

  def __init__(
      self,
      activation,
      parent_layer,
      num_bits,
      symmetric=True,
      **kwargs):
    """Construct a QuantizeAwareActivation layer.

    Args:
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      parent_layer: The layer this activation is being applied to. Such
        as Conv2D, Dense etc.
      num_bits: Number of bits for quantization
      symmetric: If true, use symmetric quantization limits instead of training
        the minimum and maximum of each quantization range separately.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    super(QuantizeAwareActivation, self).__init__(**kwargs)

    self.activation = activations.get(activation)
    self.parent_layer = parent_layer

    self.num_bits = num_bits
    self.symmetric = symmetric

    # TODO(pulkitb): Generate a meaningful name for this layer, which
    # ideally also includes the parent layer.

  def _requires_pre_quant(self):
    # TODO(pulkitb): Make this more sophisticated. This should match the
    # implementation of kernels on-device.
    return self.activation.__name__ in self._PRE_ACTIVATION_TYPES

  def build(self, input_shape):
    if self._requires_pre_quant():
      self._min_pre_activation = self.add_variable(
          'min_pre_activation',
          initializer=initializers.Constant(-6.0),
          trainable=False)
      self._max_pre_activation = self.add_variable(
          'max_pre_activation',
          initializer=initializers.Constant(6.0),
          trainable=False)

    self._min_post_activation = self.add_variable(
        'min_post_activation',
        initializer=initializers.Constant(-6.0),
        trainable=False)
    self._max_post_activation = self.add_variable(
        'max_post_activation',
        initializer=initializers.Constant(6.0),
        trainable=False)

  def call(self, inputs, training=None):
    # TODO(pulkitb): Construct graph for both training/eval modes.
    if training is None:
      training = K.learning_phase()

    x = inputs
    if self._requires_pre_quant():
      x = quant_ops.MovingAvgQuantize(
          inputs,
          self._min_pre_activation,
          self._max_pre_activation,
          ema_decay=0.999,
          is_training=training,
          num_bits=self.num_bits,
          symmetric=self.symmetric,
          name_prefix=self.name)

    x = self.activation(x)
    x = quant_ops.MovingAvgQuantize(
        x,
        self._min_post_activation,
        self._max_post_activation,
        ema_decay=0.999,
        is_training=training,
        num_bits=self.num_bits,
        symmetric=self.symmetric,
        name_prefix=self.name)

    return x

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    base_config = super(QuantizeAwareActivation, self).get_config()
    config = {
        'activation': activations.serialize(self.activation),
        'parent_layer': self.parent_layer,
        'num_bits': self.num_bits,
        'symmetric': self.symmetric,
    }
    return dict(list(base_config.items()) + list(config.items()))
