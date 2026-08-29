# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Quantization registry which specifies how layers should be quantized."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
from tensorflow_model_optimization.python.core.quantization.keras import quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.experimental.ternarization import ternarization_quantize_configs

QuantizeConfig = quantize_config.QuantizeConfig

layers = tf.keras.layers


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
    if isinstance(rnn_layer.cell, layers.StackedRNNCells):
      return rnn_layer.cell.cells
    else:
      return [rnn_layer.cell]


class TernarizationQuantizeRegistry(quantize_registry.QuantizeRegistry,
                                    _RNNHelper):
  """QuantizationRegistry for built-in Keras classes for ternarization scheme."""

  # TODO(tfmot): expand layers test in quantize_functional_test.py
  # to add more layers to allowlist.
  _LAYER_QUANTIZE_INFO = [
      # Activation Layers
      _QuantizeInfo(layers.ReLU, [], [], True),
      _QuantizeInfo(layers.Softmax, [], []),
      # Enable once verified.
      # layers.ELU,
      _QuantizeInfo(layers.LeakyReLU, [], [], True),
      # layers.PReLU,
      # layers.ThresholdedReLU,

      # Convolution Layers
      # _QuantizeInfo(layers.Conv1D, ['kernel'], ['activation']),

      # layers.Conv2D is supported and handled in code below.
      # layers.DepthwiseConv2D is supported and handled in code below.

      # _QuantizeInfo(layers.Conv3D, ['kernel'], ['activation']),
      # _QuantizeInfo(layers.Conv3DTranspose, ['kernel'], ['activation']),
      _QuantizeInfo(layers.Concatenate, [], [], True),
      _no_quantize(layers.Cropping1D),
      _no_quantize(layers.Cropping2D),
      _no_quantize(layers.Cropping3D),
      # _no_quantize(layers.UpSampling1D),

      # TODO(tfmot): Reduce the quantization errors for bilinear interpolation
      # type for UpSampling2D op. UpSampling2D supports two interpolation types,
      # nearest and bilinear. we convert the op to ResizeBilnear integer op on
      # TFLite. This ResizeBilinear TFLite op only for input and output has the
      # same quantization parameters. (scale and zero_point) To do that, The
      # TFLite converter inserts quantization cast op right after the input to
      # match quantization params for the output. Current QAT doesn’t consider
      # this behavior yet, so now we have larger quantization errors than we
      # expected. We have to add support for it on QAT or change the TFLite
      # kernel op to support different quantization params for input and output.
      # (Note that the nearest case just copies the number so there’s no more
      # errors even if the quantization order is different.)
      _QuantizeInfo(layers.UpSampling2D, [], [], True),

      # _no_quantize(layers.UpSampling3D),
      _no_quantize(layers.ZeroPadding1D),
      _no_quantize(layers.ZeroPadding2D),
      # _no_quantize(layers.ZeroPadding3D),

      # Supported via modifications in Transforms.
      # layers.SeparableConv1D, layers.SeparableConv2D,

      # Core Layers
      _no_quantize(layers.ActivityRegularization),
      _QuantizeInfo(layers.Dense, ['kernel'], ['activation']),
      _no_quantize(layers.Dropout),
      _no_quantize(layers.Flatten),
      # _no_quantize(layers.Masking),
      _no_quantize(layers.Permute),
      # _no_quantize(layers.RepeatVector),
      _no_quantize(layers.Reshape),
      _no_quantize(layers.SpatialDropout1D),
      _no_quantize(layers.SpatialDropout2D),
      _no_quantize(layers.SpatialDropout3D),
      # layers.Lambda needs custom handling by the user.

      # Pooling Layers
      _QuantizeInfo(layers.AveragePooling1D, [], [], True),
      _QuantizeInfo(layers.AveragePooling2D, [], [], True),
      # _QuantizeInfo(layers.AveragePooling3D, [], [], True),
      _QuantizeInfo(layers.GlobalAveragePooling1D, [], [], True),
      _QuantizeInfo(layers.GlobalAveragePooling2D, [], [], True),
      _QuantizeInfo(layers.GlobalAveragePooling3D, [], [], True),
      _no_quantize(layers.GlobalMaxPooling1D),
      _no_quantize(layers.GlobalMaxPooling2D),
      _no_quantize(layers.GlobalMaxPooling3D),
      # _no_quantize(layers.MaxPooling1D),
      _no_quantize(layers.MaxPooling2D),
      # _no_quantize(layers.MaxPooling3D),

      # _QuantizeInfo(layers.LocallyConnected1D, ['kernel'], ['activation']),
      # _QuantizeInfo(layers.LocallyConnected2D, ['kernel'], ['activation']),
      _QuantizeInfo(layers.Add, [], [], True),

      # Enable once verified with TFLite behavior.
      # layers.Embedding: ['embeddings'],

      # BatchNormalization is handled elsewhere, in the cases
      # where it's preceded by convolutional layers.
      #   layers.BatchNormalization: [],

      # Merge layers to be added.

      # RNN Cells
      # TODO(pulkitb): Verify RNN layers behavior.
      # TODO(tfmot): check if we still need to allowlist via compat.v1 and
      # compat.v2 to support legacy TensorFlow 2.X
      # behavior where the v2 RNN uses the v1 RNNCell instead of the v2 RNNCell.
      # See b/145939875 for details.
      # _QuantizeInfo(tf.keras.layers.GRUCell, ['kernel', 'recurrent_kernel'],
      #               ['activation', 'recurrent_activation']),
      # _QuantizeInfo(tf.keras.layers.LSTMCell, ['kernel', 'recurrent_kernel'],
      #               ['activation', 'recurrent_activation']),
      # _QuantizeInfo(tf.keras.experimental.PeepholeLSTMCell,
      #               ['kernel', 'recurrent_kernel'],
      #               ['activation', 'recurrent_activation']),
      # _QuantizeInfo(tf.keras.layers.SimpleRNNCell,
      #               ['kernel', 'recurrent_kernel'],
      #               ['activation', 'recurrent_activation']),
  ]

  def __init__(self, disable_per_axis=False):
    self._layer_quantize_map = {}
    for quantize_info in self._LAYER_QUANTIZE_INFO:
      self._layer_quantize_map[quantize_info.layer_type] = quantize_info

    self._layer_quantize_map[layers.Conv2DTranspose] = (
        ternarization_quantize_configs.TernarizationConvTransposeQuantizeConfig(
            ['kernel']))

    # For now, ternarization quantizes weight only.
    # self._disable_per_axis = disable_per_axis
    # if not self._disable_per_axis:
    #   self._layer_quantize_map[
    #       layers.
    #       Conv2D] = ternarization_quantize_configs.
    #           TernarizationConvQuantizeConfig(['kernel'])
    #   self._layer_quantize_map[
    #       layers.
    #       DepthwiseConv2D] = ternarization_quantize_configs.
    #           TernarizationConvQuantizeConfig(
    #           ['depthwise_kernel'])
    # else:
    self._layer_quantize_map[
        layers
        .Conv2D] = ternarization_quantize_configs.TernarizationQuantizeConfig(
            ['kernel'])
    self._layer_quantize_map[
        layers.
        DepthwiseConv2D] = ternarization_quantize_configs.TernarizationQuantizeConfig(
            ['depthwise_kernel'])

  def _is_supported_layer(self, layer_class):
    return layer_class in self._layer_quantize_map

  def _is_rnn_layer(self, layer):
    return layer.__class__ in {
        layers.GRU,
        layers.LSTM,
        layers.RNN,
        layers.SimpleRNN,
    }

  def _get_quantize_info(self, layer_class):
    return self._layer_quantize_map[layer_class]

  # Interface functions.

  def supports(self, layer):
    """Returns whether the registry supports this layer type.

    # TODO(pulkitb): Consider pushing this function up to the registry.

    Args:
      layer: The layer to check for support.

    Returns:
      True/False whether the layer type is supported.

    """
    if self._is_supported_layer(layer.__class__):
      return True

    if self._is_rnn_layer(layer):
      for rnn_cell in self._get_rnn_cells(layer):
        # All cells in the RNN layer should be supported.
        if not self._is_supported_layer(rnn_cell.__class__):
          return False
      return True

    return False

  def _get_quantize_config(self, layer_type):
    quantize_info = self._get_quantize_info(layer_type)

    # In case of `Activation`, there is no `_QuantizeInfo` object. It
    # directly stores a `QuantizeConfig`.
    if isinstance(quantize_info, QuantizeConfig):
      return quantize_info

    return ternarization_quantize_configs.TernarizationQuantizeConfig(
        quantize_info.weight_attrs)

  def get_quantize_config(self, layer):
    """Returns the quantization config for the given layer.

    Args:
      layer: input layer to return quantize config for.

    Returns:
      Returns the QuantizeConfig for the given layer.
    """
    if not self.supports(layer):
      raise ValueError(
          '`get_quantize_config()` called on an unsupported layer {}. Check '
          'if layer is supported by calling `supports()`. Alternatively, you '
          'can use `QuantizeConfig` to specify a behavior for your layer.'
          .format(layer.__class__))

    if self._is_supported_layer(layer.__class__):
      return self._get_quantize_config(layer.__class__)

    if self._is_rnn_layer(layer):
      weight_attrs = []
      activation_attrs = []
      for rnn_cell in self._get_rnn_cells(layer):
        quantize_info = self._get_quantize_info(rnn_cell.__class__)
        weight_attrs.append(quantize_info.weight_attrs)
        activation_attrs.append(quantize_info.activation_attrs)

      # Result quantization for RNN isn't straight-forward like regular layers.
      # To implement during full RNN support.
      return ternarization_quantize_configs.TernarizationQuantizeConfigRNN()

    # Should never come here.
    raise ValueError('Invalid Layer type {}'.format(layer.__class__))


def _types_dict():
  return {
      'TernarizationQuantizeConfig':
          ternarization_quantize_configs.TernarizationQuantizeConfig,
      'TernarizationQuantizeConfigRNN':
          ternarization_quantize_configs.TernarizationQuantizeConfigRNN,
      'NoOpQuantizeConfig':
          ternarization_quantize_configs.NoOpQuantizeConfig,
      'TernarizationOutputQuantizeConfig':
          ternarization_quantize_configs.TernarizationOutputQuantizeConfig,
      'TernarizationConvTransposeQuantizeConfig':
          ternarization_quantize_configs
          .TernarizationConvTransposeQuantizeConfig,
  }
