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
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils


class QuantizeAwareActivation(object):
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

  # These activations don't need to be quantized prior to being used.
  # `relu` gets fused into the preceding operation, and `linear` is a no-op.
  _ACTIVATIONS_NO_PRE_QUANT = {'relu', 'linear'}

  # TODO(pulkitb): Other activations such as elu, tanh etc., should just work
  # on inclusion. Verify in TFLite before enabling.
  _SUPPORTED_ACTIVATIONS = {'relu', 'softmax', 'linear'}

  _CUSTOM_ACTIVATION_ERR_MSG = (
      'Only some Keras activations under `tf.keras.activations` are supported. '
      'For other activations, use `Quantizer` directly, and update layer '
      'config using `QuantizeProvider`.'
  )

  def __init__(self, activation, quantizer, step, quantize_wrapper):
    """Construct a QuantizeAwareActivation layer.

    Args:
      activation: Activation function to use.
      quantizer: `Quantizer` to be used to quantize the activation.
      step: Variable which tracks optimizer step.
      quantize_wrapper: `QuantizeWrapper` which owns this activation.
    """
    self.activation = activation
    self.quantizer = quantizer
    self.step = step
    self.quantize_wrapper = quantize_wrapper

    if not self._is_supported_activation(self.activation):
      raise ValueError(self._CUSTOM_ACTIVATION_ERR_MSG)

    if self._should_pre_quantize():
      self._min_pre_activation, self._max_pre_activation = \
        self._add_range_weights('pre_activation')

    self._min_post_activation, self._max_post_activation = \
      self._add_range_weights('post_activation')

  def _is_supported_activation(self, activation):
    if not hasattr(activation, '__name__'):
      return False

    return activation.__name__ in self._SUPPORTED_ACTIVATIONS

  def _should_pre_quantize(self):
    return self.activation.__name__ not in self._ACTIVATIONS_NO_PRE_QUANT

  def _add_range_weights(self, name):
    min_var = self.quantize_wrapper.add_weight(
        name + '_min', initializer=initializers.Constant(-6.0), trainable=False)
    max_var = self.quantize_wrapper.add_weight(
        name + '_max', initializer=initializers.Constant(6.0), trainable=False)

    return min_var, max_var

  @property
  def training(self):
    return self._training

  @training.setter
  def training(self, value):
    self._training = value

  def _dict_vars(self, min_var, max_var):
    return {'min_var': min_var, 'max_var': max_var}

  def __call__(self, inputs, *args, **kwargs):

    def make_quantizer_fn(training, x, min_var, max_var):
      """Use currying to return True/False specialized fns to the cond."""

      def quantizer_fn(x=x,
                       quantizer=self.quantizer,
                       min_var=min_var,
                       max_var=max_var):
        return quantizer(x, self.step, training,
                         **self._dict_vars(min_var, max_var))

      return quantizer_fn

    x = inputs
    if self._should_pre_quantize():
      x = tf_utils.smart_cond(
          self._training,
          make_quantizer_fn(True, x, self._min_pre_activation,
                            self._max_pre_activation),
          make_quantizer_fn(False, x, self._min_pre_activation,
                            self._max_pre_activation))

    x = self.activation(x, *args, **kwargs)

    x = tf_utils.smart_cond(
        self._training,
        make_quantizer_fn(True, x, self._min_post_activation,
                          self._max_post_activation),
        make_quantizer_fn(False, x, self._min_post_activation,
                          self._max_post_activation))

    return x

  # `QuantizeAwareActivation` wraps the activation within a layer to perform
  # quantization. In the process, the layer's activation is replaced with
  # `QuantizeAwareActivation`.
  # However, when the layer is serialized and deserialized, we want the original
  # activation to be reconstructed. This ensures that when `QuantizeWrapper`
  # wraps the layer, it can again replace the original activation.

  @classmethod
  def from_config(cls, config):
    return activations.deserialize(config['activation'])

  def get_config(self):
    return {
        'activation': activations.serialize(self.activation)
    }
