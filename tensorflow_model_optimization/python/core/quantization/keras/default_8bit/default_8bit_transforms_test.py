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
"""Tests for Default Transforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras import quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_transforms
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm_test_utils
from tensorflow_model_optimization.python.core.quantization.keras.layers import dense_batchnorm_test_utils

ModelTransformer = model_transformer.ModelTransformer

Conv2DModel = conv_batchnorm_test_utils.Conv2DModel
DepthwiseConv2DModel = conv_batchnorm_test_utils.DepthwiseConv2DModel
DenseModel = dense_batchnorm_test_utils.DenseModel

keras = tf.keras

Conv2DBatchNormActivationQuantize = default_8bit_transforms.Conv2DBatchNormActivationQuantize
Conv2DBatchNormReLUQuantize = default_8bit_transforms.Conv2DBatchNormReLUQuantize


# TODO(alanchiao): reduce redundancy by parameterizing on Depthwise vs Conv.
class DefaultTransformsTest(tf.test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(DefaultTransformsTest, cls).setUpClass()
    np.random.seed(12345678)

  def _get_model(
      self,
      layer_type,
      squeeze_type,
      normalization_type,
      activation_type):
    activation = None
    if activation_type == 'relu':
      activation = keras.layers.ReLU(6.0)
    elif activation_type == 'act_relu':
      activation = keras.layers.Activation('relu')

    if layer_type == 'Conv2D':
      return Conv2DModel.get_nonfolded_batchnorm_model(
          model_type='functional',
          post_bn_activation=activation,
          squeeze_type=squeeze_type,
          normalization_type=normalization_type)
    elif layer_type == 'DepthwiseConv2D':
      return DepthwiseConv2DModel.get_nonfolded_batchnorm_model(
          model_type='functional',
          post_bn_activation=activation,
          squeeze_type=squeeze_type,
          normalization_type=normalization_type)
    elif layer_type == 'Dense':
      return DenseModel.get_nonfolded_batchnorm_model(
          post_bn_activation=activation, normalization_type=normalization_type)

  def _get_input_shape(self, layer_type):
    if layer_type == 'Conv2D':
      return Conv2DModel.get_batched_input_shape()
    elif layer_type == 'DepthwiseConv2D':
      return DepthwiseConv2DModel.get_batched_input_shape()
    elif layer_type == 'Dense':
      return DenseModel.get_batched_input_shape()

  def _test_conv_squeeze_or_dense_bn_activation_transform(
      self,
      layer_type,
      normalization_type,
      activation_type,
      transform_class,
      conv_or_dense_activation_class,
      normalization_quantize_config_class,
      squeeze_type=None):
    model = self._get_model(layer_type,
                            squeeze_type,
                            normalization_type,
                            activation_type)
    input_shape = self._get_input_shape(layer_type)

    transformed_model, updated_metadata = ModelTransformer(
        model,
        [transform_class()],
    ).transform()

    conv_or_dense_layer = transformed_model.layers[1]
    if squeeze_type == 'sepconv1d_squeeze':
      bn_layer = transformed_model.layers[3]
    else:
      bn_layer = transformed_model.layers[2]

    self.assertIsInstance(conv_or_dense_layer.activation,
                          conv_or_dense_activation_class)
    self.assertIsInstance(
        updated_metadata.get(bn_layer.name).get('quantize_config'),
        normalization_quantize_config_class)

    inputs = np.random.standard_normal(input_shape)
    self.assertAllClose(
        transformed_model.predict(inputs), model.predict(inputs))

  @parameterized.parameters(
      ('Conv2D', 'BatchNormalization'),
      ('Conv2D', 'SyncBatchNormalization'),
      ('DepthwiseConv2D', 'BatchNormalization'),
      ('DepthwiseConv2D', 'SyncBatchNormalization'),
  )
  def testConv2DBatchNormQuantize(self, layer_type, normalization_type):
    self._test_conv_squeeze_or_dense_bn_activation_transform(
        layer_type=layer_type,
        squeeze_type=None,
        normalization_type=normalization_type,
        activation_type=None,
        transform_class=default_8bit_transforms.Conv2DBatchNormQuantize,
        conv_or_dense_activation_class=quantize_aware_activation.NoOpActivation,
        normalization_quantize_config_class=default_8bit_quantize_configs
        .Default8BitOutputQuantizeConfig)

  @parameterized.parameters(
      ('Conv2D', 'BatchNormalization'),
      ('Conv2D', 'SyncBatchNormalization'),
      ('DepthwiseConv2D', 'BatchNormalization'),
      ('DepthwiseConv2D', 'SyncBatchNormalization'),
  )
  def testConv2DBatchNormReLUQuantize(self, layer_type, normalization_type):
    self._test_conv_squeeze_or_dense_bn_activation_transform(
        layer_type=layer_type,
        squeeze_type=None,
        normalization_type=normalization_type,
        activation_type='relu',
        transform_class=default_8bit_transforms.Conv2DBatchNormReLUQuantize,
        conv_or_dense_activation_class=quantize_aware_activation.NoOpActivation,
        normalization_quantize_config_class=default_8bit_quantize_configs
        .NoOpQuantizeConfig)

  @parameterized.parameters(
      ('Conv2D', 'BatchNormalization'),
      ('Conv2D', 'SyncBatchNormalization'),
      ('DepthwiseConv2D', 'BatchNormalization'),
      ('DepthwiseConv2D', 'SyncBatchNormalization'),
  )
  def testConv2DBatchNormActivationQuantize(
      self, layer_type, normalization_type):
    self._test_conv_squeeze_or_dense_bn_activation_transform(
        layer_type=layer_type,
        squeeze_type=None,
        normalization_type=normalization_type,
        activation_type='act_relu',
        transform_class=default_8bit_transforms
        .Conv2DBatchNormActivationQuantize,
        conv_or_dense_activation_class=quantize_aware_activation.NoOpActivation,
        normalization_quantize_config_class=default_8bit_quantize_configs
        .NoOpQuantizeConfig)

  @parameterized.parameters(
      ('Conv2D', 'BatchNormalization'),
      ('Conv2D', 'SyncBatchNormalization'),
      ('DepthwiseConv2D', 'BatchNormalization'),
      ('DepthwiseConv2D', 'SyncBatchNormalization'),
  )
  def testConv2DReshapeBatchNormQuantize(
      self, layer_type, normalization_type):
    self._test_conv_squeeze_or_dense_bn_activation_transform(
        layer_type=layer_type,
        squeeze_type='sepconv1d_squeeze',
        normalization_type=normalization_type,
        activation_type=False,
        transform_class=default_8bit_transforms.Conv2DReshapeBatchNormQuantize,
        conv_or_dense_activation_class=quantize_aware_activation.NoOpActivation,
        normalization_quantize_config_class=default_8bit_quantize_configs
        .Default8BitOutputQuantizeConfig)

  @parameterized.parameters(
      ('Conv2D', 'BatchNormalization'),
      ('Conv2D', 'SyncBatchNormalization'),
      ('DepthwiseConv2D', 'BatchNormalization'),
      ('DepthwiseConv2D', 'SyncBatchNormalization'),
  )
  def testConv2DReshapeBatchNormReLUQuantize(
      self, layer_type, normalization_type):
    self._test_conv_squeeze_or_dense_bn_activation_transform(
        layer_type=layer_type,
        squeeze_type='sepconv1d_squeeze',
        normalization_type=normalization_type,
        activation_type='relu',
        transform_class=default_8bit_transforms
        .Conv2DReshapeBatchNormReLUQuantize,
        conv_or_dense_activation_class=quantize_aware_activation.NoOpActivation,
        normalization_quantize_config_class=default_8bit_quantize_configs
        .NoOpQuantizeConfig)

  @parameterized.parameters(
      ('Conv2D', 'BatchNormalization'),
      ('Conv2D', 'SyncBatchNormalization'),
      ('DepthwiseConv2D', 'BatchNormalization'),
      ('DepthwiseConv2D', 'SyncBatchNormalization'),
  )
  def testConv2DReshapeBatchNormActivationQuantize(
      self, layer_type, normalization_type):
    self._test_conv_squeeze_or_dense_bn_activation_transform(
        layer_type=layer_type,
        squeeze_type='sepconv1d_squeeze',
        normalization_type=normalization_type,
        activation_type='act_relu',
        transform_class=default_8bit_transforms
        .Conv2DReshapeBatchNormActivationQuantize,
        conv_or_dense_activation_class=quantize_aware_activation.NoOpActivation,
        normalization_quantize_config_class=default_8bit_quantize_configs
        .NoOpQuantizeConfig)

  @parameterized.parameters(
      ('Dense', 'BatchNormalization'),
      ('Dense', 'SyncBatchNormalization'),
  )
  def testDenseBatchNormQuantize(self, layer_type, normalization_type):
    self._test_conv_squeeze_or_dense_bn_activation_transform(
        layer_type=layer_type,
        squeeze_type=None,
        normalization_type=normalization_type,
        activation_type=None,
        transform_class=default_8bit_transforms.DenseBatchNormQuantize,
        conv_or_dense_activation_class=quantize_aware_activation.NoOpActivation,
        normalization_quantize_config_class=default_8bit_quantize_configs
        .Default8BitOutputQuantizeConfig)

  @parameterized.parameters(
      ('Dense', 'BatchNormalization'),
      ('Dense', 'SyncBatchNormalization'),
  )
  def testDenseBatchNormReLUQuantize(self, layer_type, normalization_type):
    self._test_conv_squeeze_or_dense_bn_activation_transform(
        layer_type=layer_type,
        squeeze_type=None,
        normalization_type=normalization_type,
        activation_type='relu',
        transform_class=default_8bit_transforms.DenseBatchNormReLUQuantize,
        conv_or_dense_activation_class=quantize_aware_activation.NoOpActivation,
        normalization_quantize_config_class=default_8bit_quantize_configs
        .NoOpQuantizeConfig)

  @parameterized.parameters(
      ('Dense', 'BatchNormalization'),
      ('Dense', 'SyncBatchNormalization'),
  )
  def testDenseBatchNormActivationQuantize(self, layer_type,
                                           normalization_type):
    self._test_conv_squeeze_or_dense_bn_activation_transform(
        layer_type=layer_type,
        squeeze_type=None,
        normalization_type=normalization_type,
        activation_type='act_relu',
        transform_class=default_8bit_transforms
        .DenseBatchNormActivationQuantize,
        conv_or_dense_activation_class=quantize_aware_activation.NoOpActivation,
        normalization_quantize_config_class=default_8bit_quantize_configs
        .NoOpQuantizeConfig)

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'}),
      ('padding_same', {'padding': 'same'}),
      ('padding_same_dilation_2', {'padding': 'same', 'dilation_rate': 2}),
      ('strides', {'strides': 2}),
      ('dilation_rate', {'dilation_rate': 2}),
      ('depth_multiplier', {'depth_multiplier': 2}),
      ('regularizer', {
          'depthwise_regularizer': 'l2',
          'pointwise_regularizer': 'l2',
          'bias_regularizer': 'l2',
          'activity_regularizer': 'l2'}),
      ('constraint', {
          'depthwise_constraint': tf.keras.constraints.max_norm(2.),
          'pointwise_constraint': tf.keras.constraints.min_max_norm(0., 2.),
          'bias_constraint': tf.keras.constraints.unit_norm()}),
      ('activation_relu', {'activation': 'relu'}),
      # TODO(pulkitb): Temporarily disabling due to numerical errors resulting
      # from caching of activation logits in TF code.
      # ('activation_softmax', {'activation': 'softmax'}),
  )
  def testSeparableConv1DQuantize_(self, kwargs):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = 3
    num_samples = 2
    stack_size = 3
    num_row = 7

    sepconv_model = tf.keras.Sequential([
        tf.keras.Input(
            shape=(num_row, stack_size), batch_size=num_samples),
        tf.keras.layers.SeparableConv1D(**kwargs)])

    transformed_model, updated_metadata = ModelTransformer(
        sepconv_model,
        [default_8bit_transforms.SeparableConv1DQuantize()],
    ).transform()

    self.assertContainsSubset(
        {'sepconv1d_expand_1', 'separable_conv1d_QAT_SepConv2D',
         'sepconv1d_squeeze_1'},
        updated_metadata.keys())
    self.assertEqual(sepconv_model.output_shape, transformed_model.output_shape)

    x = np.random.rand(*sepconv_model.input_shape)
    y = np.random.rand(*sepconv_model.output_shape)

    # Ensure model is equivalent, and forward pass results are the same.
    self.assertAllClose(sepconv_model.predict(x), transformed_model.predict(x))

    # Ensure model is equivalent, and training results are the same.
    sepconv_model.compile(loss='categorical_crossentropy', optimizer='sgd')
    sepconv_model.fit(x, y, epochs=100)
    transformed_model.compile(loss='categorical_crossentropy', optimizer='sgd')
    transformed_model.fit(x, y, epochs=100)

    # Over a long training cycle with constraints and regularizers, the model
    # can build very minute differences. Hence reducing tol to 1e-5.
    self.assertAllClose(sepconv_model.predict(x), transformed_model.predict(x),
                        atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'}),
      ('padding_same', {'padding': 'same'}),
      ('padding_same_dilation_2',
       {'padding': 'same', 'dilation_rate': 2}),
      ('strides', {'strides': 2}),
      ('dilation_rate', {'dilation_rate': 2}),
      ('depth_multiplier', {'depth_multiplier': 2}),
      ('regularizer', {
          'depthwise_regularizer': 'l2',
          'pointwise_regularizer': 'l2',
          'bias_regularizer': 'l2',
          'activity_regularizer': 'l2'}),
      ('use_bias', {'use_bias': False}),
      ('constraint', {
          'depthwise_constraint': tf.keras.constraints.max_norm(2.),
          'pointwise_constraint': tf.keras.constraints.min_max_norm(0., 2.),
          'bias_constraint': tf.keras.constraints.unit_norm()})
  )
  def testSeparableConvQuantize_(self, kwargs):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = 3
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6

    sepconv_model = tf.keras.Sequential([
        tf.keras.Input(
            shape=(num_row, num_col, stack_size), batch_size=num_samples),
        tf.keras.layers.SeparableConv2D(**kwargs)])

    transformed_model, updated_metadata = ModelTransformer(
        sepconv_model,
        [default_8bit_transforms.SeparableConvQuantize()],
    ).transform()

    self.assertContainsSubset(
        updated_metadata.keys(), {'depthwise_conv2d', 'conv2d'})
    # Transformed model should have the same output shape
    self.assertEqual(sepconv_model.output_shape, transformed_model.output_shape)

    x = np.random.rand(*sepconv_model.input_shape)
    y = np.random.rand(*sepconv_model.output_shape)

    # Ensure model is equivalent, and forward pass results are the same.
    self.assertAllClose(sepconv_model.predict(x), transformed_model.predict(x))

    # Ensure model is equivalent, and training results are the same.
    sepconv_model.compile(loss='categorical_crossentropy', optimizer='sgd')
    transformed_model.compile(loss='categorical_crossentropy', optimizer='sgd')

    epochs = 100
    for _ in range(epochs):
      sepconv_model.fit(x, y, epochs=1, verbose=2)
      transformed_model.fit(x, y, epochs=1, verbose=2)
      self.assertAllClose(
          sepconv_model.get_weights(),
          transformed_model.get_weights())
      # To prevent accumulated numerical errors.
      transformed_model.set_weights(sepconv_model.get_weights())
      self.assertAllClose(
          sepconv_model.predict(x),
          transformed_model.predict(x))

  @parameterized.parameters(
      ('relu', default_8bit_transforms.LayerReLUQuantize),
      ('act_relu', default_8bit_transforms.LayerReluActivationQuantize),
  )
  def testAddReLUQuantize(self, activation_type, transform_type):
    add = keras.layers.Add()
    if activation_type == 'relu':
      activation = keras.layers.ReLU(6.0)
    elif activation_type == 'act_relu':
      activation = keras.layers.Activation('relu')

    inp1 = keras.layers.Input((3,))
    inp2 = keras.layers.Input((3,))
    x = activation(add([inp1, inp2]))
    model = keras.Model([inp1, inp2], x)

    transformed_model, updated_metadata = ModelTransformer(
        model,
        [transform_type()],
    ).transform()

    add_layer = transformed_model.layers[2]

    self.assertIsInstance(
        updated_metadata.get(add_layer.name).get('quantize_config'),
        default_8bit_quantize_configs.NoOpQuantizeConfig)

  @parameterized.parameters(
      ('relu', default_8bit_transforms.LayerReLUQuantize),
      ('act_relu', default_8bit_transforms.LayerReluActivationQuantize))
  def testLayerReLUQuantize(self, activation_type, transform_type):
    # TODO(b/185727342): Add tests for DepthConv and Dense
    input_shape = (3, 3, 3)
    conv_layer = tf.keras.layers.Conv2D(5, 2, input_shape=input_shape)
    if activation_type == 'relu':
      act_layer = keras.layers.ReLU(6.0)
    elif activation_type == 'act_relu':
      act_layer = keras.layers.Activation('relu')

    model = tf.keras.Sequential([conv_layer, act_layer])

    transformed_model, updated_metadata = ModelTransformer(
        model,
        [transform_type()],
    ).transform()

    self.assertIsInstance(
        updated_metadata.get(model.layers[0].name).get('quantize_config'),
        default_8bit_quantize_configs.NoOpQuantizeConfig)

    inputs = np.random.standard_normal((1,) + input_shape)
    self.assertAllClose(
        transformed_model.predict(inputs), model.predict(inputs))

  def testAddsQuantizeLayerAfterInputLayer(self):
    inp1 = keras.layers.Input((3,))
    inp2 = keras.layers.Input((3,))
    x = keras.layers.Concatenate()([inp1, inp2])
    model = keras.Model([inp1, inp2], x)

    transformed_model, _ = ModelTransformer(
        model,
        [default_8bit_transforms.InputLayerQuantize()]).transform()

    for input_layer in transformed_model._input_layers:
      layer_after_input = input_layer._outbound_nodes[0].outbound_layer
      self.assertIsInstance(
          layer_after_input,
          quantize_layer.QuantizeLayer)
      self.assertIsInstance(
          layer_after_input.quantizer, quantizers.AllValuesQuantizer)

  def testConcatTransform(self):
    r"""Tests the Concat Transform.

               Input
              /     \
         Dense       Dense
             \      /
              Concat

      One Dense layer has a pre-specified QuantizeConfig, whereas the other does
      not. The Transform should ensure both the output FakeQuants are disabled,
      and only a FakeQuant after Concat is present.
    """
    dense_1 = keras.layers.Dense(3)
    dense_2 = keras.layers.Dense(3)
    concat = keras.layers.Concatenate()

    inp = keras.layers.Input((2,))
    x1 = dense_1(inp)
    x2 = dense_2(inp)
    x = concat([x1, x2])
    model = keras.Model(inp, x)

    layer_metadata = {
        # dense_1 has an existing quantize_config.
        dense_1.name: {
            'quantize_config':
                default_8bit_quantize_configs.Default8BitOutputQuantizeConfig()
        }
    }
    _, updated_metadata = ModelTransformer(
        model, [default_8bit_transforms.ConcatTransform()],
        layer_metadata=layer_metadata).transform()

    concat_quantize_config = updated_metadata.get(
        concat.name).get('quantize_config')
    # Concat should quantize the output.
    self.assertIsInstance(
        concat_quantize_config,
        default_8bit_quantize_configs.Default8BitOutputQuantizeConfig)
    self.assertNotEmpty(concat_quantize_config.get_output_quantizers(None))

    dense_1_quantize_config = updated_metadata.get(
        dense_1.name).get('quantize_config')
    # The existing quantize_config should do nothing for outputs.
    self.assertIsInstance(
        dense_1_quantize_config,
        default_8bit_quantize_configs.Default8BitOutputQuantizeConfig)
    self.assertEmpty(dense_1_quantize_config.get_output_quantizers(None))

    dense_2_quantize_config = updated_metadata.get(
        dense_2.name).get('quantize_config')
    # The quantize_config from registry should do nothing at output.
    self.assertEqual('Default8BitQuantizeConfig',
                     dense_2_quantize_config.__class__.__name__)
    self.assertEmpty(dense_2_quantize_config.get_output_quantizers(None))

  def testConcatMultipleLevels(self):
    r"""Tests case when concats applied to concats.

            Input --------------.
           /      \      |      |
         Dense   Dense   |      |
            \    /       |      |
             Concat    Dense   Dense
                 \     /        |
                  Concat        |
                        \      /
                         Concat

    The last Concat layer should be quantized but the rest
    of the outputs should just feed into it.
    """
    inp = keras.layers.Input((3,))
    x1 = keras.layers.Dense(3)(inp)
    x2 = keras.layers.Dense(3)(inp)
    x3 = keras.layers.Dense(3)(inp)
    x4 = keras.layers.Dense(3)(inp)
    c1 = keras.layers.Concatenate()([x1, x2])
    c2 = keras.layers.Concatenate()([c1, x3])
    c3 = keras.layers.Concatenate()([c2, x4])
    model = keras.Model(inp, c3)
    model.summary()

    _, layer_metadata = ModelTransformer(
        model,
        [default_8bit_transforms.ConcatTransform()]).transform()

    for layer in model.layers[1:-1]:
      quantize_config = layer_metadata[layer.name].get('quantize_config')
      self.assertEmpty(quantize_config.get_output_quantizers(None))

    c3_layer = model.layers[-1]
    quantize_config = layer_metadata[c3_layer.name].get('quantize_config')
    self.assertIsInstance(
        quantize_config,
        default_8bit_quantize_configs.Default8BitOutputQuantizeConfig)
    self.assertNotEmpty(quantize_config.get_output_quantizers(None))

  def testConcatActivationTransform(self):
    r"""Tests the Concat Transform.

         Input  Input
          /       \
        Relu     Relu
          \       /
            Concat

      The Transform should ensure both the output FakeQuants are disabled,
      and only a FakeQuant after Concat is present.
    """
    relu_1 = keras.layers.Activation('relu')
    relu_2 = keras.layers.Activation('relu')
    concat = keras.layers.Concatenate()

    inp1 = keras.layers.Input((2,))
    inp2 = keras.layers.Input((2,))
    x1 = relu_1(inp1)
    x2 = relu_2(inp2)
    x = concat([x1, x2])
    model = keras.Model([inp1, inp2], x)

    layer_metadata = {
        # dense_1 has an existing quantize_config.
        relu_1.name: {
            'quantize_config':
                (default_8bit_quantize_registry
                 .Default8BitActivationQuantizeConfig())
        },
        relu_2.name: {
            'quantize_config':
                (default_8bit_quantize_registry
                 .Default8BitActivationQuantizeConfig())
        }
    }
    _, updated_metadata = ModelTransformer(
        model, [default_8bit_transforms.ConcatTransform()],
        layer_metadata=layer_metadata).transform()

    concat_quantize_config = updated_metadata.get(
        concat.name).get('quantize_config')
    # Concat should quantize the output.
    self.assertIsInstance(
        concat_quantize_config,
        default_8bit_quantize_configs.Default8BitOutputQuantizeConfig)
    self.assertNotEmpty(concat_quantize_config.get_output_quantizers(None))

    relu_1_quantize_config = updated_metadata.get(
        relu_1.name).get('quantize_config')
    # The existing quantize_config should do nothing for outputs.
    self.assertIsInstance(
        relu_1_quantize_config,
        default_8bit_quantize_registry.Default8BitActivationQuantizeConfig)
    self.assertEmpty(relu_1_quantize_config.get_output_quantizers(None))
    self.assertFalse(relu_1_quantize_config.quantize_output)

    relu_2_quantize_config = updated_metadata.get(
        relu_2.name).get('quantize_config')
    # The quantize_config from registry should do nothing at output.
    self.assertIsInstance(
        relu_1_quantize_config,
        default_8bit_quantize_registry.Default8BitActivationQuantizeConfig)
    self.assertEmpty(relu_2_quantize_config.get_output_quantizers(None))
    self.assertFalse(relu_2_quantize_config.quantize_output)

  def testConcatConcatTransformDisablesOutput(self):
    r"""Tests the Concat Transform.

          Input    Input   Input  Input
          Reshape Reshape Reshape Reshape
              \    /           \    /
             Concat           Concat
                   \         /
                      Concat

      The Transform should ensure all output FakeQuants are disabled,
        and only a FakeQuant after the last Concat is present.
    """
    flatten_1 = keras.layers.Flatten()
    flatten_2 = keras.layers.Flatten()
    concat_1 = keras.layers.Concatenate()
    flatten_3 = keras.layers.Flatten()
    flatten_4 = keras.layers.Flatten()
    concat_2 = keras.layers.Concatenate()
    concat = keras.layers.Concatenate()

    inp1 = keras.layers.Input((1, 2, 2))
    inp2 = keras.layers.Input((1, 2, 2))
    inp3 = keras.layers.Input((1, 2, 2))
    inp4 = keras.layers.Input((1, 2, 2))
    x1 = flatten_1(inp1)
    x2 = flatten_2(inp2)
    x3 = flatten_3(inp3)
    x4 = flatten_4(inp4)

    y1 = concat_1([x1, x2])
    y2 = concat_2([x3, x4])
    z = concat([y1, y2])
    model = keras.Model([inp1, inp2, inp3, inp4], z)
    reshapes = [flatten_1, flatten_2, flatten_3, flatten_4]
    layer_metadata = {}
    for layer in reshapes:
      layer_metadata[layer.name] = {
          'quantize_config':
              default_8bit_quantize_registry.Default8BitQuantizeConfig(
                  [], [], True)}
    _, updated_metadata = ModelTransformer(
        model, [default_8bit_transforms.ConcatTransform()],
        layer_metadata=layer_metadata).transform()

    concat_quantize_config = updated_metadata.get(
        concat.name).get('quantize_config')
    # Concat should quantize the output.
    self.assertIsInstance(
        concat_quantize_config,
        default_8bit_quantize_configs.Default8BitOutputQuantizeConfig)
    self.assertNotEmpty(concat_quantize_config.get_output_quantizers(None))

    # The existing quantize_config should do nothing for outputs.
    for layer in reshapes:
      quantize_config = updated_metadata.get(layer.name).get('quantize_config')
      self.assertIsInstance(
          quantize_config,
          default_8bit_quantize_registry.Default8BitQuantizeConfig)
      self.assertEmpty(quantize_config.get_output_quantizers(layer))
      self.assertFalse(quantize_config.quantize_output)


if __name__ == '__main__':
  if hasattr(tf.keras.__internal__, 'enable_unsafe_deserialization'):
    tf.keras.__internal__.enable_unsafe_deserialization()
  tf.test.main()
