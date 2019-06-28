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
# pylint: disable=protected-access
"""Quantization Emulation Wrapper.

   The QuantizeEmulateWrapper keras layer wrapper simulates inference time
   quantization for a particular layer of the neural network during training.
   This adds quantization loss errors during the training process and allows
   the model to learn parameters to be more robust to them during training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers.wrappers import Wrapper
from tensorflow.python.keras.utils import tf_utils
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
      raise ValueError('Unsupported Layer ' + layer.__class__)

    self._num_bits = num_bits
    self._symmetric = symmetric
    self._narrow_range = narrow_range

    self._unquantized_kernels = []
    self._quantized_kernels = []

    self._track_trackable(layer, name='layer')

    # TODO(yunluli): Work-around to handle the first layer of Sequential model
    # properly. Can remove this when it is implemented in the Wrapper base
    # class.
    # The _batch_input_shape attribute in the first layer makes a Sequential
    # model to be built. This change makes sure that when we apply the wrapper
    # to the whole model, this attribute is pulled into the wrapper to preserve
    # the 'built' state of the model.
    if not hasattr(self, '_batch_input_shape') and hasattr(
        layer, '_batch_input_shape'):
      self._batch_input_shape = self.layer._batch_input_shape

  def build(self, input_shape):
    super(QuantizeEmulateWrapper, self).build(input_shape)

    min_weights, max_weights = [], []
    # For each of the quantizable_weights, construct the necessary variables.
    # TODO(alanchiao): when validated, add per-channel as parameter, which
    # affects shape and other factors.
    for weight in self.layer.get_quantizable_weights():
      min_var = self.add_variable(
          'weight_min',
          initializer=initializers.Constant(-6.0),
          trainable=False)
      max_var = self.add_variable(
          'weight_max', initializer=initializers.Constant(6.0), trainable=False)
      self._unquantized_kernels.append(weight)
      min_weights.append(min_var)
      max_weights.append(max_var)

      # set_quantizable_weights on the wrapped layer removes unquantized_kernel
      # from _trainable_weights. We add it to the wrappers _trainable_weights
      # to ensure it gets gradient updates.
      self._trainable_weights.append(weight)

    self._weight_vars = list(
        zip(self._unquantized_kernels, min_weights, max_weights))
    self._min_activation = self.add_variable(
        'activation_min',
        initializer=initializers.Constant(-6.0),
        trainable=False)
    self._max_activation = self.add_variable(
        'activation_max',
        initializer=initializers.Constant(6.0),
        trainable=False)

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(self.layer.input_shape)

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    for unquantized_kernel, min_var, max_var in self._weight_vars:

      def last_value_quantize_fn(is_training,
                                 unquantized_kernel=unquantized_kernel,
                                 min_var=min_var,
                                 max_var=max_var):
        """Wrapper around LastValueQuantize."""

        def fn():
          return quant_ops.LastValueQuantize(
              unquantized_kernel,
              min_var,
              max_var,
              is_training=is_training,
              num_bits=self._num_bits,
              symmetric=self._symmetric,
              narrow_range=self._narrow_range,
              name_prefix=self.layer.name)

        return fn

      # Quantize the layer's weights and assign the resulting tensor to the
      # layer. This ensures the results of the forward pass use the quantized
      # tensor value. However, the internal _trainable_weights is not modified
      # since the unquantized weights need to be updated.
      quantized_kernel = tf_utils.smart_cond(training,
                                             last_value_quantize_fn(True),
                                             last_value_quantize_fn(False))

      self._quantized_kernels.append(quantized_kernel)

    self.layer.set_quantizable_weights(self._quantized_kernels)

    outputs = self.layer.call(inputs)

    if self.layer.activation is None:
      return outputs

    def moving_avg_quantize_fn(is_training):
      """Wrapper around MovingAvgQuantize."""

      def fn():
        return quant_ops.MovingAvgQuantize(
            outputs,
            self._min_activation,
            self._max_activation,
            ema_decay=0.999,
            is_training=is_training,
            num_bits=self._num_bits,
            symmetric=self._symmetric,
            narrow_range=False,
            name_prefix=self.layer.name)

      return fn

    outputs = tf_utils.smart_cond(training, moving_avg_quantize_fn(True),
                                  moving_avg_quantize_fn(False))

    return outputs

  def get_quantize_params(self):
    return {
        'num_bits': self._num_bits,
        'symmetric': self._symmetric,
        'narrow_range': self._narrow_range
    }

  def get_config(self):
    base_config = super(QuantizeEmulateWrapper, self).get_config()
    config = self.get_quantize_params()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()

    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    layer = deserialize_layer(config.pop('layer'))
    config['layer'] = layer

    return cls(**config)

  @property
  def trainable(self):
    return self.layer.trainable

  @trainable.setter
  def trainable(self, value):
    self.layer.trainable = value

  @property
  def trainable_weights(self):
    return self.layer.trainable_weights + self._trainable_weights

  @property
  def non_trainable_weights(self):
    return self.layer.non_trainable_weights + self._non_trainable_weights

  @property
  def updates(self):
    return self.layer.updates + self._updates

  @property
  def losses(self):
    return self.layer.losses + self._losses
