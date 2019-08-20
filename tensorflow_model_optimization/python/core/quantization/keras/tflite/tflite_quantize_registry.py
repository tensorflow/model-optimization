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

QuantizeProvider = quantize_provider.QuantizeProvider


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

  # Layer Attribute definitions.
  # TODO(pulkitb): Double check all attributes used are correct.

  _LAYERS_WEIGHTS_MAP = {
      layers.advanced_activations.ELU: [],
      layers.advanced_activations.LeakyReLU: [],
      layers.advanced_activations.ReLU: [],
      layers.advanced_activations.Softmax: [],
      layers.advanced_activations.ThresholdedReLU: [],
      layers.convolutional.Conv1D: ['kernel'],
      layers.convolutional.Conv2D: ['kernel'],
      layers.convolutional.Conv2DTranspose: ['kernel'],
      layers.convolutional.Conv3D: ['kernel'],
      layers.convolutional.Conv3DTranspose: ['kernel'],
      layers.convolutional.Cropping1D: [],
      layers.convolutional.Cropping2D: [],
      layers.convolutional.Cropping3D: [],
      layers.convolutional.DepthwiseConv2D: [],
      layers.convolutional.SeparableConv1D: ['pointwise_kernel'],
      layers.convolutional.SeparableConv2D: ['pointwise_kernel'],
      layers.convolutional.UpSampling1D: [],
      layers.convolutional.UpSampling2D: [],
      layers.convolutional.UpSampling3D: [],
      layers.convolutional.ZeroPadding1D: [],
      layers.convolutional.ZeroPadding2D: [],
      layers.convolutional.ZeroPadding3D: [],
      layers.core.Activation: [],
      layers.core.ActivityRegularization: [],
      layers.core.Dense: ['kernel'],
      layers.core.Dropout: [],
      layers.core.Flatten: [],
      layers.core.Lambda: [],
      layers.core.Masking: [],
      layers.core.Permute: [],
      layers.core.RepeatVector: [],
      layers.core.Reshape: [],
      layers.core.SpatialDropout1D: [],
      layers.core.SpatialDropout2D: [],
      layers.core.SpatialDropout3D: [],
      layers.embeddings.Embedding: ['embeddings'],
      layers.local.LocallyConnected1D: ['kernel'],
      layers.local.LocallyConnected2D: ['kernel'],
      layers.merge.Add: [],
      layers.merge.Average: [],
      layers.merge.Concatenate: [],
      layers.merge.Dot: [],
      layers.merge.Maximum: [],
      layers.merge.Minimum: [],
      layers.merge.Multiply: [],
      layers.merge.Subtract: [],
      layers.noise.AlphaDropout: [],
      layers.noise.GaussianDropout: [],
      layers.noise.GaussianNoise: [],
      layers.normalization.BatchNormalization: [],
      layers.normalization.LayerNormalization: [],
      layers.pooling.AveragePooling1D: [],
      layers.pooling.AveragePooling2D: [],
      layers.pooling.AveragePooling3D: [],
      layers.pooling.GlobalAveragePooling1D: [],
      layers.pooling.GlobalAveragePooling2D: [],
      layers.pooling.GlobalAveragePooling3D: [],
      layers.pooling.GlobalMaxPooling1D: [],
      layers.pooling.GlobalMaxPooling2D: [],
      layers.pooling.GlobalMaxPooling3D: [],
      layers.pooling.MaxPooling1D: [],
      layers.pooling.MaxPooling2D: [],
      layers.pooling.MaxPooling3D: [],
  }

  _LAYERS_ACTIVATIONS_MAP = {
      layers.advanced_activations.ELU: [],
      layers.advanced_activations.LeakyReLU: [],
      layers.advanced_activations.ReLU: [],
      layers.advanced_activations.Softmax: [],
      layers.advanced_activations.ThresholdedReLU: [],
      layers.convolutional.Conv1D: ['activation'],
      layers.convolutional.Conv2D: ['activation'],
      layers.convolutional.Conv2DTranspose: ['activation'],
      layers.convolutional.Conv3D: ['activation'],
      layers.convolutional.Conv3DTranspose: ['activation'],
      layers.convolutional.Cropping1D: [],
      layers.convolutional.Cropping2D: [],
      layers.convolutional.Cropping3D: [],
      layers.convolutional.DepthwiseConv2D: [],
      layers.convolutional.SeparableConv1D: ['activation'],
      layers.convolutional.SeparableConv2D: ['activation'],
      layers.convolutional.UpSampling1D: [],
      layers.convolutional.UpSampling2D: [],
      layers.convolutional.UpSampling3D: [],
      layers.convolutional.ZeroPadding1D: [],
      layers.convolutional.ZeroPadding2D: [],
      layers.convolutional.ZeroPadding3D: [],
      layers.core.Activation: [],
      layers.core.ActivityRegularization: [],
      layers.core.Dense: ['activation'],
      layers.core.Dropout: [],
      layers.core.Flatten: [],
      layers.core.Lambda: [],
      layers.core.Masking: [],
      layers.core.Permute: [],
      layers.core.RepeatVector: [],
      layers.core.Reshape: [],
      layers.core.SpatialDropout1D: [],
      layers.core.SpatialDropout2D: [],
      layers.core.SpatialDropout3D: [],
      layers.embeddings.Embedding: [],
      layers.local.LocallyConnected1D: ['activation'],
      layers.local.LocallyConnected2D: ['activation'],
      layers.merge.Add: [],
      layers.merge.Average: [],
      layers.merge.Concatenate: [],
      layers.merge.Dot: [],
      layers.merge.Maximum: [],
      layers.merge.Minimum: [],
      layers.merge.Multiply: [],
      layers.merge.Subtract: [],
      layers.noise.AlphaDropout: [],
      layers.noise.GaussianDropout: [],
      layers.noise.GaussianNoise: [],
      layers.normalization.BatchNormalization: [],
      layers.normalization.LayerNormalization: [],
      layers.pooling.AveragePooling1D: [],
      layers.pooling.AveragePooling2D: [],
      layers.pooling.AveragePooling3D: [],
      layers.pooling.GlobalAveragePooling1D: [],
      layers.pooling.GlobalAveragePooling2D: [],
      layers.pooling.GlobalAveragePooling3D: [],
      layers.pooling.GlobalMaxPooling1D: [],
      layers.pooling.GlobalMaxPooling2D: [],
      layers.pooling.GlobalMaxPooling3D: [],
      layers.pooling.MaxPooling1D: [],
      layers.pooling.MaxPooling2D: [],
      layers.pooling.MaxPooling3D: [],
  }

  _RNN_CELLS_WEIGHTS_MAP = {
      layers.recurrent.GRUCell: ['kernel', 'recurrent_kernel'],
      layers.recurrent.LSTMCell: ['kernel', 'recurrent_kernel'],
      layers.recurrent.PeepholeLSTMCell: ['kernel', 'recurrent_kernel'],
      layers.recurrent.SimpleRNNCell: ['kernel', 'recurrent_kernel'],
  }

  _RNN_CELLS_ACTIVATIONS_MAP = {
      layers.recurrent.GRUCell: ['activation', 'recurrent_activation'],
      layers.recurrent.LSTMCell: ['activation', 'recurrent_activation'],
      layers.recurrent.PeepholeLSTMCell: ['activation', 'recurrent_activation'],
      layers.recurrent.SimpleRNNCell: ['activation'],
  }

  _RNN_LAYERS = {
      layers.recurrent.GRU,
      layers.recurrent.LSTM,
      layers.recurrent.RNN,
      layers.recurrent.SimpleRNN,
  }

  # Support functions.

  def _is_supported_non_rnn_layer(self, layer):
    return layer.__class__ in self._LAYERS_WEIGHTS_MAP

  def _is_supported_rnn_layer(self, layer):
    return layer.__class__ in self._RNN_LAYERS

  def _is_supported_rnn_cell(self, layer):
    return layer.__class__ in self._RNN_CELLS_WEIGHTS_MAP

  def _weight_attrs(self, layer):
    return self._LAYERS_WEIGHTS_MAP[layer.__class__]

  def _activation_attrs(self, layer):
    return self._LAYERS_ACTIVATIONS_MAP[layer.__class__]

  def _weight_attrs_rnn_cell(self, cell):
    return self._RNN_CELLS_WEIGHTS_MAP[cell.__class__]

  def _activation_attrs_rnn_cell(self, cell):
    return self._RNN_CELLS_ACTIVATIONS_MAP[cell.__class__]

  # Interface functions.

  def supports(self, layer):
    """Returns whether the registry supports this layer type.

    # TODO(pulkitb): Consider pushing this function up to the registry.

    Args:
      layer: The layer to check for support.

    Returns:
      True/False whether the layer type is supported.

    """
    if self._is_supported_non_rnn_layer(layer):
      return True

    if self._is_supported_rnn_layer(layer):
      for rnn_cell in self._get_rnn_cells(layer):
        # All cells in the RNN layer should be supported. It's possible to use
        # custom cells in an RNN layer.
        if not self._is_supported_rnn_cell(rnn_cell):
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

    if self._is_supported_non_rnn_layer(layer):
      return TFLiteQuantizeProvider(
          self._weight_attrs(layer), self._activation_attrs(layer))

    if self._is_supported_rnn_layer(layer):
      weight_attrs = []
      activation_attrs = []
      for rnn_cell in self._get_rnn_cells(layer):
        weight_attrs.append(self._weight_attrs_rnn_cell(rnn_cell))
        activation_attrs.append(self._activation_attrs_rnn_cell(rnn_cell))

      return TFLiteQuantizeProviderRNN(weight_attrs, activation_attrs)

    # Should never come here.
    raise ValueError('Invalid Layer type {}'.format(layer.__class__))


class TFLiteQuantizeProvider(QuantizeProvider):
  """QuantizeProvider for non recurrent Keras layers."""

  def __init__(self, weight_attrs, activation_attrs):
    self.weight_attrs = weight_attrs
    self.activation_attrs = activation_attrs

    # TODO(pulkitb): For some layers such as Conv2D, per_axis should be True.
    # Add mapping for which layers support per_axis.
    self.weight_quantizer = quantizers.LastValueQuantizer(
        num_bits=8, per_axis=False, symmetric=True)
    self.activation_quantizer = quantizers.MovingAverageQuantizer(
        num_bits=8, per_axis=False, symmetric=True)

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
