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
"""Quantization registry which specifies how layers should be quantized.

Module: tfmot.quantization.keras.tflite
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import layers

from tensorflow_model_optimization.python.core.quantization.keras import quantize_provider
from tensorflow_model_optimization.python.core.quantization.keras import quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm
from tensorflow_model_optimization.python.core.quantization.keras.tflite import tflite_quantize_providers
from tensorflow_model_optimization.python.core.quantization.keras.tflite import tflite_quantizers

QuantizeProvider = quantize_provider.QuantizeProvider


class _QuantizeInfo(object):
  """QuantizeInfo."""

  def __init__(self,
               layer_type,
               weight_attrs,
               activation_attrs,
               quantize_output=False):
    """QuantizeInfo.

    Args:
      layer_type: Type of keras layer.
      weight_attrs: List of quantizable weight attributes of layer.
      activation_attrs: List of quantizable activation attributes of layer.
      quantize_output: Bool. Should we quantize the output of the layer.
    """
    self.layer_type = layer_type
    self.weight_attrs = weight_attrs
    self.activation_attrs = activation_attrs
    self.quantize_output = quantize_output


def _no_quantize(layer_type):
  return _QuantizeInfo(layer_type, [], [], False)


class _RNNHelper(object):
  """Helper functions for working with RNNs."""

  def _get_rnn_cells(self, rnn_layer):
    """Returns the list of cells in an RNN layer."""
    if isinstance(rnn_layer.cell, layers.recurrent.StackedRNNCells):
      return rnn_layer.cell.cells
    else:
      return [rnn_layer.cell]


class TFLiteQuantizeRegistry(quantize_registry.QuantizeRegistry, _RNNHelper):
  """QuantizationRegistry for built-in Keras classes for TFLite scheme."""

  _LAYER_QUANTIZE_INFO = [

      # Activation Layers
      _QuantizeInfo(layers.advanced_activations.ReLU, [], [], True),
      _QuantizeInfo(layers.advanced_activations.Softmax, [], []),
      # Enable once verified.
      # layers.advanced_activations.ELU,
      # layers.advanced_activations.LeakyReLU,
      # layers.advanced_activations.PReLU,
      # layers.advanced_activations.ThresholdedReLU,

      # Convolution Layers
      _QuantizeInfo(layers.convolutional.Conv1D, ['kernel'], ['activation']),
      _QuantizeInfo(layers.convolutional.Conv3D, ['kernel'], ['activation']),
      # TODO(pulkitb): Verify Transpose layers.
      _QuantizeInfo(layers.convolutional.Conv2DTranspose,
                    ['kernel'], ['activation']),
      _QuantizeInfo(layers.convolutional.Conv3DTranspose,
                    ['kernel'], ['activation']),
      _no_quantize(layers.convolutional.Cropping1D),
      _no_quantize(layers.convolutional.Cropping2D),
      _no_quantize(layers.convolutional.Cropping3D),
      _no_quantize(layers.convolutional.UpSampling1D),
      _no_quantize(layers.convolutional.UpSampling2D),
      _no_quantize(layers.convolutional.UpSampling3D),
      _no_quantize(layers.convolutional.ZeroPadding1D),
      _no_quantize(layers.convolutional.ZeroPadding2D),
      _no_quantize(layers.convolutional.ZeroPadding3D),
      # Enable once verified.
      # layers.convolutional.SeparableConv1D,
      # layers.convolutional.SeparableConv2D,

      # Core Layers
      _no_quantize(layers.core.ActivityRegularization),
      _QuantizeInfo(layers.core.Dense, ['kernel'], ['activation']),
      _no_quantize(layers.core.Dropout),
      _no_quantize(layers.core.Flatten),
      _no_quantize(layers.core.Masking),
      _no_quantize(layers.core.Permute),
      _no_quantize(layers.core.RepeatVector),
      _no_quantize(layers.core.Reshape),
      _no_quantize(layers.core.SpatialDropout1D),
      _no_quantize(layers.core.SpatialDropout2D),
      _no_quantize(layers.core.SpatialDropout3D),
      # layers.core.Lambda needs custom handling by the user.

      # Pooling Layers
      _QuantizeInfo(layers.pooling.AveragePooling1D, [], [], True),
      _QuantizeInfo(layers.pooling.AveragePooling2D, [], [], True),
      _QuantizeInfo(layers.pooling.AveragePooling3D, [], [], True),
      _QuantizeInfo(layers.pooling.GlobalAveragePooling1D, [], [], True),
      _QuantizeInfo(layers.pooling.GlobalAveragePooling2D, [], [], True),
      _QuantizeInfo(layers.pooling.GlobalAveragePooling3D, [], [], True),
      _no_quantize(layers.pooling.GlobalMaxPooling1D),
      _no_quantize(layers.pooling.GlobalMaxPooling2D),
      _no_quantize(layers.pooling.GlobalMaxPooling3D),
      _no_quantize(layers.pooling.MaxPooling1D),
      _no_quantize(layers.pooling.MaxPooling2D),
      _no_quantize(layers.pooling.MaxPooling3D),

      # TODO(pulkitb): Verify Locally Connected layers.
      _QuantizeInfo(layers.local.LocallyConnected1D,
                    ['kernel'], ['activation']),
      _QuantizeInfo(layers.local.LocallyConnected2D,
                    ['kernel'], ['activation']),

      # Enable once verified with TFLite behavior.
      # layers.embeddings.Embedding: ['embeddings'],
      # layers.normalization.BatchNormalization: [],

      # Merge layers to be added.

      # RNN Cells
      # TODO(pulkitb): Verify RNN layers behavior.
      _QuantizeInfo(layers.recurrent.GRUCell, ['kernel', 'recurrent_kernel'],
                    ['activation', 'recurrent_activation']),
      _QuantizeInfo(layers.recurrent.LSTMCell, ['kernel', 'recurrent_kernel'],
                    ['activation', 'recurrent_activation']),
      _QuantizeInfo(layers.recurrent.PeepholeLSTMCell,
                    ['kernel', 'recurrent_kernel'],
                    ['activation', 'recurrent_activation']),
      _QuantizeInfo(layers.recurrent.SimpleRNNCell,
                    ['kernel', 'recurrent_kernel'],
                    ['activation', 'recurrent_activation']),

      # TODO(tf-mot): Move layers out once Transforms indicate quantization.
      _no_quantize(conv_batchnorm._ConvBatchNorm2D),  # pylint: disable=protected-access
      _no_quantize(conv_batchnorm._DepthwiseConvBatchNorm2D),  # pylint: disable=protected-access
  ]

  def __init__(self):
    self._layer_quantize_map = {}
    for quantize_info in self._LAYER_QUANTIZE_INFO:
      self._layer_quantize_map[quantize_info.layer_type] = quantize_info

    # Hack for `Activation` layer. That is the only layer with a separate
    # QuantizeProvider.
    self._layer_quantize_map[layers.Activation] = ActivationQuantizeProvider()
    self._layer_quantize_map[layers.Conv2D] = ConvQuantizeProvider(
        ['kernel'], ['activation'], False)
    self._layer_quantize_map[layers.DepthwiseConv2D] = ConvQuantizeProvider(
        ['depthwise_kernel'], ['activation'], False)

  def _is_supported_layer(self, layer):
    return layer.__class__ in self._layer_quantize_map

  def _is_rnn_layer(self, layer):
    return layer.__class__ in {
        layers.recurrent.GRU,
        layers.recurrent.LSTM,
        layers.recurrent.RNN,
        layers.recurrent.SimpleRNN,
    }

  def _get_quantize_info(self, layer):
    return self._layer_quantize_map[layer.__class__]

  # Interface functions.

  def supports(self, layer):
    """Returns whether the registry supports this layer type.

    # TODO(pulkitb): Consider pushing this function up to the registry.

    Args:
      layer: The layer to check for support.

    Returns:
      True/False whether the layer type is supported.

    """
    if self._is_supported_layer(layer):
      return True

    if self._is_rnn_layer(layer):
      for rnn_cell in self._get_rnn_cells(layer):
        # All cells in the RNN layer should be supported.
        if not self._is_supported_layer(rnn_cell):
          return False
      return True

    return False

  def get_quantize_provider(self, layer):
    """Returns the quantization provider for the given layer.

    Args:
      layer: input layer to return quantize provider for.

    Returns:
      Returns the QuantizeProvider for the given layer.
    """
    if not self.supports(layer):
      raise ValueError(
          '`get_quantize_provider()` called on an unsupported layer {}. Check '
          'if layer is supported by calling `supports()`. Alternatively, you '
          'can use `QuantizeProvider` to specify a behavior for your layer.'
          .format(layer.__class__))

    if self._is_supported_layer(layer):
      quantize_info = self._get_quantize_info(layer)

      # In case of `Activation`, there is no `_QuantizeInfo` object. It
      # directly stores a `QuantizeProvider`.
      if isinstance(quantize_info, QuantizeProvider):
        return quantize_info

      return TFLiteQuantizeProvider(
          quantize_info.weight_attrs, quantize_info.activation_attrs,
          quantize_info.quantize_output)

    if self._is_rnn_layer(layer):
      weight_attrs = []
      activation_attrs = []
      for rnn_cell in self._get_rnn_cells(layer):
        quantize_info = self._get_quantize_info(rnn_cell)
        weight_attrs.append(quantize_info.weight_attrs)
        activation_attrs.append(quantize_info.activation_attrs)

      # Result quantization for RNN isn't straight-forward like regular layers.
      # To implement during full RNN support.
      return TFLiteQuantizeProviderRNN(weight_attrs, activation_attrs, False)

    # Should never come here.
    raise ValueError('Invalid Layer type {}'.format(layer.__class__))


class TFLiteQuantizeProvider(QuantizeProvider):
  """QuantizeProvider for non recurrent Keras layers."""

  def __init__(self, weight_attrs, activation_attrs, quantize_output):
    self.weight_attrs = weight_attrs
    self.activation_attrs = activation_attrs
    self.quantize_output = quantize_output

    # TODO(pulkitb): For some layers such as Conv2D, per_axis should be True.
    # Add mapping for which layers support per_axis.
    self.weight_quantizer = quantizers.LastValueQuantizer(
        num_bits=8, per_axis=False, symmetric=True, narrow_range=True)
    self.activation_quantizer = quantizers.MovingAverageQuantizer(
        num_bits=8, per_axis=False, symmetric=False, narrow_range=False)

  def get_weights_and_quantizers(self, layer):
    return [(getattr(layer, weight_attr), self.weight_quantizer)
            for weight_attr in self.weight_attrs]

  def get_activations_and_quantizers(self, layer):
    return [(getattr(layer, activation_attr), self.activation_quantizer)
            for activation_attr in self.activation_attrs]

  def set_quantize_weights(self, layer, quantize_weights):
    if len(self.weight_attrs) != len(quantize_weights):
      raise ValueError(
          '`set_quantize_weights` called on layer {} with {} '
          'weight parameters, but layer expects {} values.'.format(
              layer.name, len(quantize_weights), len(self.weight_attrs)))

    for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
      current_weight = getattr(layer, weight_attr)
      if current_weight.shape != weight.shape:
        raise ValueError('Existing layer weight shape {} is incompatible with'
                         'provided weight shape {}'.format(
                             current_weight.shape, weight.shape))

      setattr(layer, weight_attr, weight)

  def set_quantize_activations(self, layer, quantize_activations):
    if len(self.activation_attrs) != len(quantize_activations):
      raise ValueError(
          '`set_quantize_activations` called on layer {} with {} '
          'activation parameters, but layer expects {} values.'.format(
              layer.name, len(quantize_activations),
              len(self.activation_attrs)))

    for activation_attr, activation in \
        zip(self.activation_attrs, quantize_activations):
      setattr(layer, activation_attr, activation)

  def get_output_quantizers(self, layer):
    if self.quantize_output:
      return [self.activation_quantizer]
    return []

  @classmethod
  def from_config(cls, config):
    """Instantiates a `TFLiteQuantizeProvider` from its config.

    Args:
        config: Output of `get_config()`.

    Returns:
        A `TFLiteQuantizeProvider` instance.
    """
    return cls(**config)

  def get_config(self):
    # TODO(pulkitb): Add weight and activation quantizer to config.
    # Currently it's created internally, but ideally the quantizers should be
    # part of the constructor and passed in from the registry.
    return {
        'weight_attrs': self.weight_attrs,
        'activation_attrs': self.activation_attrs,
        'quantize_output': self.quantize_output
    }

  def __eq__(self, other):
    if not isinstance(other, TFLiteQuantizeProvider):
      return False

    return (self.weight_attrs == other.weight_attrs and
            self.activation_attrs == self.activation_attrs and
            self.weight_quantizer == other.weight_quantizer and
            self.activation_quantizer == other.activation_quantizer and
            self.quantize_output == other.quantize_output)

  def __ne__(self, other):
    return not self.__eq__(other)


class TFLiteQuantizeProviderRNN(TFLiteQuantizeProvider, _RNNHelper):
  """QuantizeProvider for RNN layers."""

  def get_weights_and_quantizers(self, layer):
    weights_quantizers = []
    for weight_attrs_cell, rnn_cell in \
        zip(self.weight_attrs, self._get_rnn_cells(layer)):
      for weight_attr in weight_attrs_cell:
        weights_quantizers.append(
            (getattr(rnn_cell, weight_attr), self.weight_quantizer))

    return weights_quantizers

  def get_activations_and_quantizers(self, layer):
    activations_quantizers = []
    for activation_attrs_cell, rnn_cell in \
        zip(self.activation_attrs, self._get_rnn_cells(layer)):
      for activation_attr in activation_attrs_cell:
        activations_quantizers.append(
            (getattr(rnn_cell, activation_attr), self.activation_quantizer))

    return activations_quantizers

  def _flatten(self, list_of_lists):
    flat_list = []
    for sublist in list_of_lists:
      for item in sublist:
        flat_list.append(item)
    return flat_list

  def set_quantize_weights(self, layer, quantize_weights):
    flattened_weight_attrs = self._flatten(self.weight_attrs)
    if len(flattened_weight_attrs) != len(quantize_weights):
      raise ValueError(
          '`set_quantize_weights` called on layer {} with {} '
          'weight parameters, but layer expects {} values.'.format(
              layer.name, len(quantize_weights), len(flattened_weight_attrs)))

    i = 0
    for weight_attrs_cell, rnn_cell in \
        zip(self.weight_attrs, self._get_rnn_cells(layer)):
      for weight_attr in weight_attrs_cell:
        current_weight = getattr(rnn_cell, weight_attr)
        quantize_weight = quantize_weights[i]

        if current_weight.shape != quantize_weight.shape:
          raise ValueError('Existing layer weight shape {} is incompatible with'
                           'provided weight shape {}'.format(
                               current_weight.shape, quantize_weight.shape))

        setattr(rnn_cell, weight_attr, quantize_weight)
        i += 1

  def set_quantize_activations(self, layer, quantize_activations):
    flattened_activation_attrs = self._flatten(self.activation_attrs)
    if len(flattened_activation_attrs) != len(quantize_activations):
      raise ValueError(
          '`set_quantize_activations` called on layer {} with {} '
          'activation parameters, but layer expects {} values.'.format(
              layer.name, len(quantize_activations),
              len(flattened_activation_attrs)))

    i = 0
    for activation_attrs_cell, rnn_cell in \
        zip(self.activation_attrs, self._get_rnn_cells(layer)):
      for activation_attr in activation_attrs_cell:
        setattr(rnn_cell, activation_attr, quantize_activations[i])
        i += 1


class ActivationQuantizeProvider(QuantizeProvider):
  """QuantizeProvider for keras.layers.Activation.

  `keras.layers.Activation` needs a separate `QuantizeProvider` since the
  decision to quantize depends on the specific activation type.
  """

  def _assert_activation_layer(self, layer):
    if not isinstance(layer, layers.Activation):
      raise RuntimeError('ActivationQuantizeProvider can only be used with '
                         '`keras.layers.Activation`.')

  def get_weights_and_quantizers(self, layer):
    self._assert_activation_layer(layer)
    return []

  def get_activations_and_quantizers(self, layer):
    self._assert_activation_layer(layer)
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    self._assert_activation_layer(layer)

  def set_quantize_activations(self, layer, quantize_activations):
    self._assert_activation_layer(layer)

  def get_output_quantizers(self, layer):
    self._assert_activation_layer(layer)

    if not hasattr(layer.activation, '__name__'):
      raise ValueError('Activation {} not supported by '
                       'ActivationQuantizeProvider.'.format(layer.activation))

    if layer.activation.__name__ in ['relu']:
      # 'relu' should generally get fused into the previous layer.
      return [quantizers.MovingAverageQuantizer(
          num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]
    elif layer.activation.__name__ in ['linear', 'softmax']:
      return []

    raise ValueError('Activation {} not supported by '
                     'ActivationQuantizeProvider.'.format(layer.activation))

  def get_config(self):
    return {}


class ConvQuantizeProvider(TFLiteQuantizeProvider):
  """QuantizeProvider for Conv2D/DepthwiseConv2D layers."""

  def __init__(self, weight_attrs, activation_attrs, quantize_output):
    super(ConvQuantizeProvider, self).__init__(
        weight_attrs, activation_attrs, quantize_output)

    self.weight_quantizer = tflite_quantizers.ConvWeightsQuantizer()


def _types_dict():
  return {
      'TFLiteQuantizeProvider': TFLiteQuantizeProvider,
      'TFLiteQuantizeProviderRNN': TFLiteQuantizeProviderRNN,
      'ActivationQuantizeProvider': ActivationQuantizeProvider,
      'ConvQuantizeProvider': ConvQuantizeProvider,
      'NoOpQuantizeProvider': tflite_quantize_providers.NoOpQuantizeProvider,
      'OutputQuantizeProvider': tflite_quantize_providers.OutputQuantizeProvider
  }
