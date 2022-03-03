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
"""Default 8-bit transforms."""

import collections
import inspect

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend

from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras import quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms

LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern

keras = tf.keras


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
  # custom QuantizeConfig.
  layer_metadata = {'quantize_config': None}

  return LayerNode(layer_config, weights, metadata=layer_metadata)


def _get_quantize_config(layer_node):
  return layer_node.metadata.get('quantize_config')


def _has_custom_quantize_config(*layer_nodes):
  for layer_node in layer_nodes:
    if _get_quantize_config(layer_node) is not None:
      return True
  return False


def _normalize_tuple(value):
  if isinstance(value, int):
    return (value,)
  else:
    return tuple(value)


class Conv2DBatchNormQuantize(transforms.Transform):
  """Transform to be applied to "Conv2D" + "BatchNorm" Graph.

  This transform disables Quantization between Conv and BatchNorm
  to ensure FQ does not get placed between them.
  """

  def pattern(self):
    return LayerPattern(
        'BatchNormalization|SyncBatchNormalization',
        inputs=[LayerPattern(
            'Conv2D|DepthwiseConv2D', config={'activation': 'linear'})])

  def _replace(self, bn_layer_node, conv_layer_node):
    if _has_custom_quantize_config(bn_layer_node, conv_layer_node):
      return bn_layer_node

    conv_layer_node.layer['config']['activation'] = (
        keras.activations.serialize(quantize_aware_activation.NoOpActivation()))
    bn_layer_node.metadata['quantize_config'] = (
        default_8bit_quantize_configs.Default8BitOutputQuantizeConfig())

    return bn_layer_node

  def replacement(self, match_layer):
    bn_layer_node = match_layer
    conv_layer_node = match_layer.input_layers[0]

    return self._replace(bn_layer_node, conv_layer_node)

  def custom_objects(self):
    return {
        'NoOpQuantizeConfig': default_8bit_quantize_configs.NoOpQuantizeConfig,
        'NoOpActivation': quantize_aware_activation.NoOpActivation
    }


class Conv2DReshapeBatchNormQuantize(Conv2DBatchNormQuantize):
  """Transform to be applied to "Conv2D" + "Reshape" + "BatchNorm" Graph.

  This transform disables Quantization between Conv, Reshape and BatchNorm
  to ensure FQ does not get placed between them.
  """

  def pattern(self):
    return LayerPattern(
        'BatchNormalization|SyncBatchNormalization',
        inputs=[LayerPattern(
            'Lambda', config={'name': 'sepconv1d_squeeze.*'},
            inputs=[LayerPattern(
                'Conv2D|DepthwiseConv2D',
                config={'activation': 'linear'})])])

  def replacement(self, match_layer):
    bn_layer_node = match_layer
    reshape_layer_node = bn_layer_node.input_layers[0]
    conv_layer_node = reshape_layer_node.input_layers[0]

    return self._replace(bn_layer_node, conv_layer_node)


class Conv2DBatchNormReLUQuantize(Conv2DBatchNormQuantize):
  """Transform to be applied to "Conv2D" + "BatchNorm" + "ReLU" Graph.

  This transform disables Quantization between Conv, BatchNorm and ReLU
  to ensure FQ does not get placed between them.
  """

  def pattern(self):
    return LayerPattern(
        # TODO(pulkitb): Enhance match to only occur for relu, relu1 and relu6
        'ReLU',
        inputs=[super(Conv2DBatchNormReLUQuantize, self).pattern()])

  def _replace(self, relu_layer_node, bn_layer_node, conv_layer_node):
    if _has_custom_quantize_config(
        relu_layer_node, bn_layer_node, conv_layer_node):
      return relu_layer_node

    conv_layer_node.layer['config']['activation'] = (
        keras.activations.serialize(quantize_aware_activation.NoOpActivation()))
    bn_layer_node.metadata['quantize_config'] = (
        default_8bit_quantize_configs.NoOpQuantizeConfig())

    return relu_layer_node

  def replacement(self, match_layer):
    relu_layer_node = match_layer
    bn_layer_node = relu_layer_node.input_layers[0]
    conv_layer_node = bn_layer_node.input_layers[0]

    return self._replace(relu_layer_node, bn_layer_node, conv_layer_node)


class Conv2DBatchNormActivationQuantize(Conv2DBatchNormReLUQuantize):
  """Transform to be applied to "Conv2D" + "BatchNorm" + "ReLU" Graph.

  This transform disables Quantization between Conv, BatchNorm and ReLU
  to ensure FQ does not get placed between them.
  """

  def pattern(self):
    return LayerPattern(
        'Activation',
        config={'activation': 'relu'},
        inputs=[Conv2DBatchNormQuantize.pattern(self)])


class Conv2DReshapeBatchNormReLUQuantize(Conv2DBatchNormReLUQuantize):
  """Transform to be applied to "Conv2D" + "Reshape" + "BatchNorm" + "ReLU" Graph.

  This transform disables Quantization between Conv, Reshape, BatchNorm and ReLU
  to ensure FQ does not get placed between them.
  """

  def pattern(self):
    return LayerPattern(
        'ReLU',
        inputs=[Conv2DReshapeBatchNormQuantize.pattern(self)])

  def replacement(self, match_layer):
    relu_layer_node = match_layer
    bn_layer_node = relu_layer_node.input_layers[0]
    squeeze_layer_node = bn_layer_node.input_layers[0]
    conv_layer_node = squeeze_layer_node.input_layers[0]

    return self._replace(relu_layer_node, bn_layer_node, conv_layer_node)


class Conv2DReshapeBatchNormActivationQuantize(
    Conv2DReshapeBatchNormReLUQuantize):
  """Transform to be applied to "Conv2D" + "Reshape" + "BatchNorm" + "ReLU" Graph.

  This transform disables Quantization between Conv, Reshape, BatchNorm and ReLU
  to ensure FQ does not get placed between them.
  """

  def pattern(self):
    return LayerPattern(
        'Activation',
        config={'activation': 'relu'},
        inputs=[Conv2DReshapeBatchNormQuantize.pattern(self)])


class DenseBatchNormQuantize(transforms.Transform):
  """Transform to be applied to "Dense"+ "BatchNorm" Graph.

  This transform disables Quantization between Dense and BatchNorm
  to ensure FQ does not get placed between them.
  """

  def pattern(self):
    return LayerPattern(
        'BatchNormalization|SyncBatchNormalization',
        inputs=[LayerPattern('Dense', config={'activation': 'linear'})])

  def _replace(self, bn_layer_node, dense_layer_node):
    if _has_custom_quantize_config(bn_layer_node, dense_layer_node):
      return bn_layer_node

    dense_layer_node.layer['config']['activation'] = (
        keras.activations.serialize(quantize_aware_activation.NoOpActivation()))
    bn_layer_node.metadata['quantize_config'] = (
        default_8bit_quantize_configs.Default8BitOutputQuantizeConfig())

    return bn_layer_node

  def replacement(self, match_layer):
    bn_layer_node = match_layer
    dense_layer_node = match_layer.input_layers[0]

    return self._replace(bn_layer_node, dense_layer_node)

  def custom_objects(self):
    return {
        'NoOpQuantizeConfig': default_8bit_quantize_configs.NoOpQuantizeConfig,
        'NoOpActivation': quantize_aware_activation.NoOpActivation
    }


class DenseBatchNormReLUQuantize(DenseBatchNormQuantize):
  """Transform to be applied to "Dense"+ "BatchNorm" + "ReLU" Graph.

  This transform disables Quantization between Dense, BatchNorm and ReLU
  to ensure FQ does not get placed between them.
  """

  def pattern(self):
    return LayerPattern(
        'ReLU', inputs=[super(DenseBatchNormReLUQuantize, self).pattern()])

  def _replace(self, relu_layer_node, bn_layer_node, dense_layer_node):
    if _has_custom_quantize_config(relu_layer_node, bn_layer_node,
                                   dense_layer_node):
      return relu_layer_node

    dense_layer_node.layer['config']['activation'] = (
        keras.activations.serialize(quantize_aware_activation.NoOpActivation()))
    bn_layer_node.metadata['quantize_config'] = (
        default_8bit_quantize_configs.NoOpQuantizeConfig())

    return relu_layer_node

  def replacement(self, match_layer):
    relu_layer_node = match_layer
    bn_layer_node = relu_layer_node.input_layers[0]
    dense_layer_node = bn_layer_node.input_layers[0]

    return self._replace(relu_layer_node, bn_layer_node, dense_layer_node)


class DenseBatchNormActivationQuantize(DenseBatchNormReLUQuantize):
  """Transform to be applied to "Dense"+ "BatchNorm" + "ReLU" Graph.

  This transform disables Quantization between Dense, BatchNorm and ReLU
  to ensure FQ does not get placed between them.
  """

  def pattern(self):
    return LayerPattern(
        'Activation',
        config={'activation': 'relu'},
        inputs=[DenseBatchNormQuantize.pattern(self)])


class SeparableConv1DQuantize(transforms.Transform):
  """Add QAT support for Keras SeparableConv1D layer.

  Transforms SeparableConv1D into a SeparableConv2D invocation. The Keras
  SeparableConv1D layer internally uses the same code as a SeparbaleConv2D
  layer. It simple expands and squeezes the tensor dimensions before and after
  the convolutions. Applying this transform ensures the QAT handling for
  SeparableConv2D kicks in and handles the FQ placement properly.

  Maps:
  Input -> SeparableConv1D -> Output
    to
  Input -> Lambda(ExpandDims) -> SeparableConv2D -> Lambda(Squeeze) -> Output

  Unlike SeparableConv2DQuantize, this does not break the layer into
  DepthwiseConv and Conv separately, since no DepthwiseConv1D exists.
  """

  def pattern(self):
    return LayerPattern('SeparableConv1D')

  def _get_name(self, prefix):
    # TODO(pulkitb): Move away from `backend.unique_object_name` since it isn't
    # exposed as externally usable.
    return backend.unique_object_name(prefix)

  def replacement(self, match_layer):
    if _has_custom_quantize_config(match_layer):
      return match_layer

    sepconv1d_layer = match_layer.layer
    sepconv1d_config = sepconv1d_layer['config']
    sepconv1d_weights = list(match_layer.weights.values())

    padding = sepconv1d_config['padding']
    # SepConv2D does not accept causal padding, and SepConv1D has some special
    # handling for it.
    # TODO(pulkitb): Add support for causal padding.
    if padding == 'causal':
      raise ValueError('SeparableConv1D with causal padding is not supported.')

    # TODO(pulkitb): Handle other base_layer args such as dtype, input_dim etc.

    sepconv2d_layer = tf.keras.layers.SeparableConv2D(
        filters=sepconv1d_config['filters'],
        kernel_size=(1,) + _normalize_tuple(sepconv1d_config['kernel_size']),
        strides=_normalize_tuple(sepconv1d_config['strides']) * 2,
        padding=padding,
        data_format=sepconv1d_config['data_format'],
        dilation_rate=(1,) + _normalize_tuple(
            sepconv1d_config['dilation_rate']),
        depth_multiplier=sepconv1d_config['depth_multiplier'],
        activation=sepconv1d_config['activation'],
        use_bias=sepconv1d_config['use_bias'],
        depthwise_initializer=sepconv1d_config['depthwise_initializer'],
        pointwise_initializer=sepconv1d_config['pointwise_initializer'],
        bias_initializer=sepconv1d_config['bias_initializer'],
        depthwise_regularizer=sepconv1d_config['depthwise_regularizer'],
        pointwise_regularizer=sepconv1d_config['pointwise_regularizer'],
        bias_regularizer=sepconv1d_config['bias_regularizer'],
        activity_regularizer=sepconv1d_config['activity_regularizer'],
        depthwise_constraint=sepconv1d_config['depthwise_constraint'],
        pointwise_constraint=sepconv1d_config['pointwise_constraint'],
        bias_constraint=sepconv1d_config['bias_constraint'],
        # TODO(pulkitb): Rethink what to do for name. Using the same name leads
        # to confusion, since it's typically separable_conv1d
        name=sepconv1d_config['name'] + '_QAT_SepConv2D',
        trainable=sepconv1d_config['trainable']
    )

    sepconv2d_weights = collections.OrderedDict()
    sepconv2d_weights['depthwise_kernel:0'] = np.expand_dims(
        sepconv1d_weights[0], 0)
    sepconv2d_weights['pointwise_kernel:0'] = np.expand_dims(
        sepconv1d_weights[1], 0)
    if sepconv1d_config['use_bias']:
      sepconv2d_weights['bias:0'] = sepconv1d_weights[2]

    if sepconv1d_config['data_format'] == 'channels_last':
      spatial_dim = 1
    else:
      spatial_dim = 2

    sepconv2d_layer_config = keras.layers.serialize(sepconv2d_layer)
    sepconv2d_layer_config['name'] = sepconv2d_layer.name

    # Needed to ensure these new layers are considered for quantization.
    sepconv2d_metadata = {'quantize_config': None}

    # TODO(pulkitb): Consider moving from Lambda to custom ExpandDims/Squeeze.

    # Layer before SeparableConv2D which expands input tensors to match 2D.
    expand_layer = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x, spatial_dim),
        name=self._get_name('sepconv1d_expand'))
    expand_layer_config = keras.layers.serialize(expand_layer)
    expand_layer_config['name'] = expand_layer.name
    expand_layer_metadata = {
        'quantize_config': default_8bit_quantize_configs.NoOpQuantizeConfig()}

    squeeze_layer = tf.keras.layers.Lambda(
        lambda x: tf.squeeze(x, [spatial_dim]),
        name=self._get_name('sepconv1d_squeeze'))
    squeeze_layer_config = keras.layers.serialize(squeeze_layer)
    squeeze_layer_config['name'] = squeeze_layer.name
    squeeze_layer_metadata = {
        'quantize_config': default_8bit_quantize_configs.NoOpQuantizeConfig()}

    return LayerNode(
        squeeze_layer_config,
        metadata=squeeze_layer_metadata,
        input_layers=[LayerNode(
            sepconv2d_layer_config,
            weights=sepconv2d_weights,
            metadata=sepconv2d_metadata,
            input_layers=[LayerNode(
                expand_layer_config, metadata=expand_layer_metadata)]
            )])


class SeparableConvQuantize(transforms.Transform):
  """Break SeparableConv into a DepthwiseConv and Conv layer.

  SeparableConv is a composition of a DepthwiseConv and a Conv layer. For the
  purpose of quantization, a FQ operation needs to be placed between the output
  of DepthwiseConv and the following Conv.

  This is needed since there is a dynamic tensor in between the two layers, and
  it's range information needs to be captured by the FakeQuant op to ensure
  full int8 quantization of the layers is possible.

  Splitting the layer into 2 ensures that each individual layer is handled
  correctly with respect to quantization.
  """

  def pattern(self):
    return LayerPattern('SeparableConv2D')

  def replacement(self, match_layer):
    if _has_custom_quantize_config(match_layer):
      return match_layer

    sepconv_layer = match_layer.layer
    sepconv_weights = list(match_layer.weights.values())

    # TODO(pulkitb): SeparableConv has kwargs other than constructor args which
    # need to be handled.
    # Applicable to both layers: trainable, dtype, name
    # Applicable to dconv: input_dim, input_shape, batch_input_shape, batch_size
    # Needs special handling: weights
    # Unknown: dynamic, autocast

    dconv_layer = tf.keras.layers.DepthwiseConv2D(
        kernel_size=sepconv_layer['config']['kernel_size'],
        strides=sepconv_layer['config']['strides'],
        padding=sepconv_layer['config']['padding'],
        depth_multiplier=sepconv_layer['config']['depth_multiplier'],
        data_format=sepconv_layer['config']['data_format'],
        dilation_rate=sepconv_layer['config']['dilation_rate'],
        activation=None,
        use_bias=False,
        depthwise_initializer=sepconv_layer['config']['depthwise_initializer'],
        depthwise_regularizer=sepconv_layer['config']['depthwise_regularizer'],
        depthwise_constraint=sepconv_layer['config']['depthwise_constraint'],
        trainable=sepconv_layer['config']['trainable']
    )
    dconv_weights = collections.OrderedDict()
    dconv_weights['depthwise_kernel:0'] = sepconv_weights[0]
    dconv_layer_config = keras.layers.serialize(dconv_layer)
    dconv_layer_config['name'] = dconv_layer.name
    # Needed to ensure these new layers are considered for quantization.
    dconv_metadata = {'quantize_config': None}

    conv_layer = tf.keras.layers.Conv2D(
        filters=sepconv_layer['config']['filters'],
        kernel_size=(1, 1),  # (1,) * rank
        strides=(1, 1),
        padding='valid',
        data_format=sepconv_layer['config']['data_format'],
        dilation_rate=sepconv_layer['config']['dilation_rate'],
        groups=1,
        activation=sepconv_layer['config']['activation'],
        use_bias=sepconv_layer['config']['use_bias'],
        kernel_initializer=sepconv_layer['config']['pointwise_initializer'],
        bias_initializer=sepconv_layer['config']['bias_initializer'],
        kernel_regularizer=sepconv_layer['config']['pointwise_regularizer'],
        bias_regularizer=sepconv_layer['config']['bias_regularizer'],
        activity_regularizer=sepconv_layer['config']['activity_regularizer'],
        kernel_constraint=sepconv_layer['config']['pointwise_constraint'],
        bias_constraint=sepconv_layer['config']['bias_constraint'],
        trainable=sepconv_layer['config']['trainable']
    )
    conv_weights = collections.OrderedDict()
    conv_weights['kernel:0'] = sepconv_weights[1]
    if sepconv_layer['config']['use_bias']:
      conv_weights['bias:0'] = sepconv_weights[2]
    conv_layer_config = keras.layers.serialize(conv_layer)
    conv_layer_config['name'] = conv_layer.name
    # Needed to ensure these new layers are considered for quantization.
    conv_metadata = {'quantize_config': None}

    dconv_layer_node = LayerNode(
        dconv_layer_config, weights=dconv_weights, metadata=dconv_metadata)
    return LayerNode(
        conv_layer_config,
        weights=conv_weights,
        input_layers=[dconv_layer_node],
        metadata=conv_metadata)


class LayerReLUQuantize(transforms.Transform):
  """Transform to be applied to "Add"+ "ReLU" Graph.

  This transform disables Quantization between Add and ReLU
  to ensure FQ does not get placed between them.
  """

  def pattern(self):
    return LayerPattern(
        'ReLU', inputs=[LayerPattern('Add|Conv2D|DepthwiseConv2D|Dense')])

  def replacement(self, match_layer):
    relu_layer_node = match_layer
    add_layer_node = relu_layer_node.input_layers[0]

    add_layer_node.metadata['quantize_config'] = (
        default_8bit_quantize_configs.NoOpQuantizeConfig())

    return match_layer

  def custom_objects(self):
    return {
        'NoOpQuantizeConfig': default_8bit_quantize_configs.NoOpQuantizeConfig,
    }


class LayerReluActivationQuantize(LayerReLUQuantize):
  """Transform to be applied to "Add"+ "ReLU" Graph.

  This transform disables Quantization between Add and ReLU
  to ensure FQ does not get placed between them.
  """

  def pattern(self):
    return LayerPattern(
        'Activation',
        config={'activation': 'relu'},
        inputs=[LayerPattern('Add|Conv2D|DepthwiseConv2D|Dense')])


class InputLayerQuantize(transforms.Transform):
  """Quantizes InputLayer, by adding QuantizeLayer after it.

  InputLayer => InputLayer -> QuantizeLayer
  """

  def pattern(self):
    return LayerPattern('InputLayer')

  def replacement(self, match_layer):
    quant_layer = quantize_layer.QuantizeLayer(
        quantizers.AllValuesQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False))
    layer_config = keras.layers.serialize(quant_layer)
    layer_config['name'] = quant_layer.name

    quant_layer_node = LayerNode(
        layer_config,
        input_layers=[match_layer])

    return quant_layer_node

  def custom_objects(self):
    return {
        'QuantizeLayer': quantize_layer.QuantizeLayer,
        'MovingAverageQuantizer': quantizers.MovingAverageQuantizer,
        'AllValuesQuantizer': quantizers.AllValuesQuantizer
    }


class ConcatTransform(transforms.Transform):
  """Transform for Concatenate. Quantize only after concatenation."""

  # pylint:disable=protected-access

  def pattern(self):
    # TODO(pulkitb): Write a clean way to handle arbitrary length patterns.
    return LayerPattern(
        'Concatenate', inputs=[LayerPattern('.*'), LayerPattern('.*')])

  def _get_layer_type(self, layer_class_name):
    if layer_class_name == 'QuantizeLayer':
      return quantize_layer.QuantizeLayer
    keras_layers = inspect.getmembers(tf.keras.layers, inspect.isclass)
    for layer_name, layer_type in keras_layers:
      if layer_name == layer_class_name:
        return layer_type
    return None

  def _disable_output_quantize(self, quantize_config):
    # TODO(pulkitb): Disabling quantize_config may also require handling
    # activation quantizers. Handle that properly.
    if hasattr(quantize_config, 'quantize_output'):
      quantize_config.quantize_output = False

    quantize_config.get_output_quantizers = lambda layer: []

  def replacement(self, match_layer):
    concat_layer_node = match_layer
    feeding_layer_nodes = match_layer.input_layers

    default_registry = (
        default_8bit_quantize_registry.Default8BitQuantizeRegistry())

    feed_quantize_configs = []
    for feed_layer_node in feeding_layer_nodes:
      quantize_config = feed_layer_node.metadata.get('quantize_config')
      if not quantize_config:
        layer_class = self._get_layer_type(feed_layer_node.layer['class_name'])
        if layer_class is None:
          # Concat has an input layer we don't recognize. Return.
          return match_layer

        if layer_class == keras.layers.Concatenate:
          # Input layer to Concat is also Concat. Don't quantize it.
          feed_layer_node.metadata['quantize_config'] = (
              default_8bit_quantize_configs.NoOpQuantizeConfig())
          continue

        if layer_class == quantize_layer.QuantizeLayer:
          feed_layer_node.metadata['quantizer'] = None
          continue

        if not default_registry._is_supported_layer(layer_class):
          # Feeding layer is not supported by registry
          return match_layer

        quantize_config = default_registry._get_quantize_config(layer_class)
        feed_layer_node.metadata['quantize_config'] = quantize_config

      feed_quantize_configs.append(quantize_config)

    # TODO(pulkitb): this currently only disables output quantize config, but
    # cannot properly handle if the FQ was added to the activation. Hand this
    # properly.
    for quantize_config in feed_quantize_configs:
      self._disable_output_quantize(quantize_config)

    if not concat_layer_node.metadata.get('quantize_config'):
      concat_layer_node.metadata['quantize_config'] = (
          default_8bit_quantize_configs.Default8BitOutputQuantizeConfig())

    return concat_layer_node

  # pylint:enable=protected-access


class ConcatTransform3Inputs(ConcatTransform):

  def pattern(self):
    return LayerPattern(
        'Concatenate',
        inputs=[LayerPattern('.*'), LayerPattern('.*'), LayerPattern('.*')])


class ConcatTransform4Inputs(ConcatTransform):

  def pattern(self):
    return LayerPattern(
        'Concatenate',
        inputs=[LayerPattern('.*'), LayerPattern('.*'), LayerPattern('.*'),
                LayerPattern('.*')])


class ConcatTransform5Inputs(ConcatTransform):

  def pattern(self):
    return LayerPattern(
        'Concatenate',
        inputs=[LayerPattern('.*'), LayerPattern('.*'), LayerPattern('.*'),
                LayerPattern('.*'), LayerPattern('.*')])


class ConcatTransform6Inputs(ConcatTransform):

  def pattern(self):
    return LayerPattern(
        'Concatenate',
        inputs=[LayerPattern('.*'), LayerPattern('.*'), LayerPattern('.*'),
                LayerPattern('.*'), LayerPattern('.*'), LayerPattern('.*')])
