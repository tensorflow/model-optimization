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
"""TFLite transforms."""

import collections

from tensorflow.python import keras

from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm
from tensorflow_model_optimization.python.core.quantization.keras.tflite import tflite_quantize_providers

LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern

_ConvBatchNorm2D = conv_batchnorm._ConvBatchNorm2D  # pylint: disable=protected-access
_DepthwiseConvBatchNorm2D = conv_batchnorm._DepthwiseConvBatchNorm2D  # pylint: disable=protected-access


def _get_conv_bn_layers(bn_layer_node):
  bn_layer = bn_layer_node.layer
  conv_layer = bn_layer_node.input_layers[0].layer
  return conv_layer, bn_layer


def _get_weights(bn_layer_node):
  """Returns weight values for fused layer, including copying original values in unfused version."""

  return collections.OrderedDict(
      list(bn_layer_node.input_layers[0].weights.items())
      + list(bn_layer_node.weights.items()))


def _get_params(conv_layer, bn_layer, relu_layer=None):
  """Retrieve conv_bn params within wrapped layers."""
  if 'use_bias' in conv_layer['config']:
    if conv_layer['config']['use_bias']:
      raise ValueError(
          'use_bias should not be set to True in a Conv layer when followed '
          'by BatchNormalization. The bias in the Conv would be redundant '
          'with the one in the BatchNormalization.')

    del conv_layer['config']['use_bias']

  if 'name' in bn_layer['config']:
    del bn_layer['config']['name']

  # TODO(pulkitb): remove key conflicts
  params = dict(
      list(conv_layer['config'].items()) + list(bn_layer['config'].items()))

  if relu_layer is not None:
    params['post_activation'] = keras.layers.deserialize(relu_layer)

  return params


def _get_layer_node(fused_layer, weights):
  layer_config = keras.layers.serialize(fused_layer)
  layer_config['name'] = layer_config['config']['name']
  # This config tracks which layers get quantized, and whether they have a
  # custom QuantizeProvider.
  layer_metadata = {'quantize_provider': None}

  return LayerNode(layer_config, weights, metadata=layer_metadata)


class Conv2DBatchNormFold(transforms.Transform):
  """Conv2DBatchNormFold."""

  def pattern(self):
    return LayerPattern('BatchNormalization', {},
                        [LayerPattern('Conv2D', {}, [])])

  def replacement(self, match_layer):
    conv_layer, bn_layer = _get_conv_bn_layers(match_layer)

    fused_params = _get_params(conv_layer, bn_layer)
    fused_layer = _ConvBatchNorm2D(**fused_params)

    weights = _get_weights(match_layer)
    return _get_layer_node(fused_layer, weights)

  def custom_objects(self):
    return {'_ConvBatchNorm2D': _ConvBatchNorm2D}


class Conv2DBatchNormReLU6Fold(Conv2DBatchNormFold):
  """Conv2DBatchNormReLU6Fold."""

  def pattern(self):
    return LayerPattern('ReLU', {'max_value': 6}, [
        LayerPattern('BatchNormalization', {},
                     [LayerPattern('Conv2D', {}, [])])
    ])

  def replacement(self, match_layer):
    relu_layer = match_layer.layer
    conv_layer, bn_layer = _get_conv_bn_layers(match_layer.input_layers[0])

    fused_params = _get_params(conv_layer, bn_layer, relu_layer)
    fused_layer = _ConvBatchNorm2D(**fused_params)

    weights = _get_weights(match_layer.input_layers[0])
    return _get_layer_node(fused_layer, weights)


class DepthwiseConv2DBatchNormReLU6Fold(transforms.Transform):
  """DepthwiseConv2DBatchNormReLU6Fold."""

  def pattern(self):
    return LayerPattern('ReLU', {'max_value': 6}, [
        LayerPattern('BatchNormalization', {},
                     [LayerPattern('DepthwiseConv2D', {}, [])])
    ])

  def replacement(self, match_layer):
    relu_layer = match_layer.layer
    conv_layer, bn_layer = _get_conv_bn_layers(match_layer.input_layers[0])

    fused_params = _get_params(conv_layer, bn_layer, relu_layer)
    fused_layer = _DepthwiseConvBatchNorm2D(**fused_params)

    weights = _get_weights(match_layer.input_layers[0])
    return _get_layer_node(fused_layer, weights)

  def custom_objects(self):
    return {'_DepthwiseConvBatchNorm2D': _DepthwiseConvBatchNorm2D}


class Conv2DBatchNormQuantize(transforms.Transform):
  """Ensure FQ does not get placed between Conv and BatchNorm."""

  def pattern(self):
    return LayerPattern(
        'BatchNormalization',
        inputs=[LayerPattern(
            'Conv2D|DepthwiseConv2D', config={'activation': 'linear'})])

  @staticmethod
  def _get_quantize_provider(layer_node):
    return layer_node.metadata.get('quantize_provider')

  def _has_custom_quantize_provider(self, *layer_nodes):
    for layer_node in layer_nodes:
      if self._get_quantize_provider(layer_node) is not None:
        return True
    return False

  def replacement(self, match_layer):
    bn_layer_node, conv_layer_node = match_layer, match_layer.input_layers[0]

    if self._has_custom_quantize_provider(bn_layer_node, conv_layer_node):
      return match_layer

    conv_layer_node.layer['config']['activation'] = \
      keras.activations.serialize(quantize_aware_activation.NoOpActivation())
    bn_layer_node.metadata['quantize_provider'] = \
      tflite_quantize_providers.OutputQuantizeProvider()

    return match_layer

  def custom_objects(self):
    return {
        'tflite_quantize_providers':
            tflite_quantize_providers.OutputQuantizeProvider,
        'NoOpActivation': quantize_aware_activation.NoOpActivation
    }


class Conv2DBatchNormReLUQuantize(Conv2DBatchNormQuantize):
  """Ensure FQ does not get placed between Conv, BatchNorm and ReLU."""

  def pattern(self):
    return LayerPattern(
        # TODO(pulkitb): Enhance match to only occur for relu, relu1 and relu6
        'ReLU',
        inputs=[super(Conv2DBatchNormReLUQuantize, self).pattern()])

  def replacement(self, match_layer):
    relu_layer_node = match_layer
    bn_layer_node = relu_layer_node.input_layers[0]
    conv_layer_node = bn_layer_node.input_layers[0]

    if self._has_custom_quantize_provider(
        relu_layer_node, bn_layer_node, conv_layer_node):
      return match_layer

    conv_layer_node.layer['config']['activation'] = \
      keras.activations.serialize(quantize_aware_activation.NoOpActivation())
    bn_layer_node.metadata['quantize_provider'] = \
      tflite_quantize_providers.NoOpQuantizeProvider()

    return match_layer

  def custom_objects(self):
    return {
        'tflite_quantize_providers':
            tflite_quantize_providers.NoOpQuantizeProvider,
        'NoOpActivation': quantize_aware_activation.NoOpActivation
    }
