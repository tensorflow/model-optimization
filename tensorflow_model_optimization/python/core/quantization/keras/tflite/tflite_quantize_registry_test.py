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
"""Tests for TFLiteQuantizeRegistry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers as l
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.platform import test

from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.tflite import tflite_quantize_registry


class _TestHelper(object):

  def _convert_list(self, list_of_tuples):
    """Transforms a list of 2-tuples to a tuple of 2 lists.

    `QuantizeProvider` methods return a list of 2-tuples in the form
    [(weight1, quantizer1), (weight2, quantizer2)]. This function converts
    it into a 2-tuple of lists. ([weight1, weight2]), (quantizer1, quantizer2).

    Args:
      list_of_tuples: List of 2-tuples.

    Returns:
      2-tuple of lists.
    """
    list1 = []
    list2 = []
    for a, b in list_of_tuples:
      list1.append(a)
      list2.append(b)

    return list1, list2

  # TODO(pulkitb): Consider asserting on full equality for quantizers.

  def _assert_weight_quantizers(self, quantizer_list):
    for quantizer in quantizer_list:
      self.assertIsInstance(quantizer, quantizers.LastValueQuantizer)

  def _assert_activation_quantizers(self, quantizer_list):
    for quantizer in quantizer_list:
      self.assertIsInstance(quantizer, quantizers.MovingAverageQuantizer)

  def _assert_kernel_equality(self, a, b):
    # TODO(tfmot): branch on eager mode also.
    if tf.__version__[0] == '1':
      self.assertEqual(a, b)
    # TODO(tfmot): find why self.assertAllEqual(a, b) doesn't work as
    # https://www.tensorflow.org/api_docs/python/tf/test/TestCase#assertAllClose
    # suggests).
    else:
      self.assertAllEqual(a.numpy(), b.numpy())


class TFLiteQuantizeRegistryTest(test.TestCase, _TestHelper):

  def setUp(self):
    super(TFLiteQuantizeRegistryTest, self).setUp()
    self.quantize_registry = tflite_quantize_registry.TFLiteQuantizeRegistry()

  class CustomLayer(l.Layer):
    pass

  # supports() tests.

  def testSupports_KerasLayer(self):
    self.assertTrue(self.quantize_registry.supports(l.Dense(10)))
    self.assertTrue(self.quantize_registry.supports(l.Conv2D(10, (2, 2))))

  def testSupports_KerasRNNLayers(self):
    self.assertTrue(self.quantize_registry.supports(l.LSTM(10)))
    self.assertTrue(self.quantize_registry.supports(l.GRU(10)))

  def testSupports_KerasRNNLayerWithKerasRNNCells(self):
    self.assertTrue(self.quantize_registry.supports(l.RNN(cell=l.LSTMCell(10))))
    self.assertTrue(
        self.quantize_registry.supports(
            l.RNN(cell=[l.LSTMCell(10), l.GRUCell(10)])))

  def testDoesNotSupport_CustomLayer(self):
    self.assertFalse(self.quantize_registry.supports(self.CustomLayer()))

  def testDoesNotSupport_RNNLayerWithCustomRNNCell(self):

    class MinimalRNNCell(l.Layer):

      def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    self.assertFalse(
        self.quantize_registry.supports(l.RNN(cell=MinimalRNNCell(10))))
    self.assertFalse(
        self.quantize_registry.supports(
            l.RNN(cell=[l.LSTMCell(10), MinimalRNNCell(10)])))

  # get_quantize_provider() tests.

  def testRaisesError_UnsupportedLayer(self):
    with self.assertRaises(ValueError):
      self.quantize_registry.get_quantize_provider(self.CustomLayer())

  def testReturnsProvider_KerasLayer(self):
    model = keras.Sequential([(
        l.Dense(2, input_shape=(3,)))])
    layer = model.layers[0]

    quantize_provider = self.quantize_registry.get_quantize_provider(layer)

    (weights, weight_quantizers) = self._convert_list(
        quantize_provider.get_weights_and_quantizers(layer))
    (activations, activation_quantizers) = self._convert_list(
        quantize_provider.get_activations_and_quantizers(layer))

    self._assert_weight_quantizers(weight_quantizers)
    self.assertEqual([layer.kernel], weights)

    self._assert_activation_quantizers(activation_quantizers)
    self.assertEqual([layer.activation], activations)

    quantize_kernel = keras.backend.variable(
        np.ones(layer.kernel.shape.as_list()))
    quantize_activation = keras.activations.relu
    quantize_provider.set_quantize_weights(layer, [quantize_kernel])
    quantize_provider.set_quantize_activations(layer, [quantize_activation])

    self._assert_kernel_equality(layer.kernel, quantize_kernel)
    self.assertEqual(layer.activation, quantize_activation)

  def testReturnsProvider_LayerWithResultQuantizer(self):
    layer = l.ReLU()
    quantize_provider = self.quantize_registry.get_quantize_provider(layer)

    output_quantizers = quantize_provider.get_output_quantizers(layer)

    self.assertLen(output_quantizers, 1)
    self._assert_activation_quantizers(output_quantizers)

  def testReturnsProvider_KerasRNNLayer(self):
    model = keras.Sequential([(
        l.LSTM(2, input_shape=(3, 2)))])
    layer = model.layers[0]

    quantize_provider = self.quantize_registry.get_quantize_provider(layer)

    (weights, weight_quantizers) = self._convert_list(
        quantize_provider.get_weights_and_quantizers(layer))
    (activations, activation_quantizers) = self._convert_list(
        quantize_provider.get_activations_and_quantizers(layer))

    self._assert_weight_quantizers(weight_quantizers)
    self.assertEqual([layer.cell.kernel, layer.cell.recurrent_kernel], weights)

    self._assert_activation_quantizers(activation_quantizers)
    self.assertEqual(
        [layer.cell.activation, layer.cell.recurrent_activation], activations)

  def testReturnsProvider_KerasRNNLayerWithKerasRNNCells(self):
    lstm_cell = l.LSTMCell(3)
    gru_cell = l.GRUCell(2)
    model = keras.Sequential([l.RNN([lstm_cell, gru_cell], input_shape=(3, 2))])
    layer = model.layers[0]

    quantize_provider = self.quantize_registry.get_quantize_provider(layer)

    (weights, weight_quantizers) = self._convert_list(
        quantize_provider.get_weights_and_quantizers(layer))
    (activations, activation_quantizers) = self._convert_list(
        quantize_provider.get_activations_and_quantizers(layer))

    self._assert_weight_quantizers(weight_quantizers)
    self.assertEqual(
        [lstm_cell.kernel, lstm_cell.recurrent_kernel,
         gru_cell.kernel, gru_cell.recurrent_kernel],
        weights)

    self._assert_activation_quantizers(activation_quantizers)
    self.assertEqual(
        [lstm_cell.activation, lstm_cell.recurrent_activation,
         gru_cell.activation, gru_cell.recurrent_activation],
        activations)

  def testReturnsActivationProvider_Activation(self):
    activation_layer = keras.layers.Activation('relu')

    quantize_provider = self.quantize_registry.get_quantize_provider(
        activation_layer)

    self.assertIsInstance(
        quantize_provider, tflite_quantize_registry.ActivationQuantizeProvider)
    self._assert_activation_quantizers(
        quantize_provider.get_output_quantizers(activation_layer))


class TFLiteQuantizeProviderTest(test.TestCase, _TestHelper):

  def _simple_dense_layer(self):
    layer = l.Dense(2)
    layer.build(input_shape=(3,))
    return layer

  def testGetsQuantizeWeightsAndQuantizers(self):
    layer = self._simple_dense_layer()

    quantize_provider = tflite_quantize_registry.TFLiteQuantizeProvider(
        ['kernel'], ['activation'], False)
    (weights, weight_quantizers) = self._convert_list(
        quantize_provider.get_weights_and_quantizers(layer))

    self._assert_weight_quantizers(weight_quantizers)
    self.assertEqual([layer.kernel], weights)

  def testGetsQuantizeActivationsAndQuantizers(self):
    layer = self._simple_dense_layer()

    quantize_provider = tflite_quantize_registry.TFLiteQuantizeProvider(
        ['kernel'], ['activation'], False)
    (activations, activation_quantizers) = self._convert_list(
        quantize_provider.get_activations_and_quantizers(layer))

    self._assert_activation_quantizers(activation_quantizers)
    self.assertEqual([layer.activation], activations)

  def testSetsQuantizeWeights(self):
    layer = self._simple_dense_layer()
    quantize_kernel = K.variable(np.ones(layer.kernel.shape.as_list()))

    quantize_provider = tflite_quantize_registry.TFLiteQuantizeProvider(
        ['kernel'], ['activation'], False)
    quantize_provider.set_quantize_weights(layer, [quantize_kernel])

    self._assert_kernel_equality(layer.kernel, quantize_kernel)

  def testSetsQuantizeActivations(self):
    layer = self._simple_dense_layer()
    quantize_activation = keras.activations.relu

    quantize_provider = tflite_quantize_registry.TFLiteQuantizeProvider(
        ['kernel'], ['activation'], False)
    quantize_provider.set_quantize_activations(layer, [quantize_activation])

    self.assertEqual(layer.activation, quantize_activation)

  def testSetsQuantizeWeights_ErrorOnWrongNumberOfWeights(self):
    layer = self._simple_dense_layer()
    quantize_kernel = K.variable(np.ones(layer.kernel.shape.as_list()))

    quantize_provider = tflite_quantize_registry.TFLiteQuantizeProvider(
        ['kernel'], ['activation'], False)

    with self.assertRaises(ValueError):
      quantize_provider.set_quantize_weights(layer, [])

    with self.assertRaises(ValueError):
      quantize_provider.set_quantize_weights(
          layer, [quantize_kernel, quantize_kernel])

  def testSetsQuantizeWeights_ErrorOnWrongShapeOfWeight(self):
    layer = self._simple_dense_layer()
    quantize_kernel = K.variable(np.ones([1, 2]))

    quantize_provider = tflite_quantize_registry.TFLiteQuantizeProvider(
        ['kernel'], ['activation'], False)

    with self.assertRaises(ValueError):
      quantize_provider.set_quantize_weights(layer, [quantize_kernel])

  def testSetsQuantizeActivations_ErrorOnWrongNumberOfActivations(self):
    layer = self._simple_dense_layer()
    quantize_activation = keras.activations.relu

    quantize_provider = tflite_quantize_registry.TFLiteQuantizeProvider(
        ['kernel'], ['activation'], False)

    with self.assertRaises(ValueError):
      quantize_provider.set_quantize_activations(layer, [])

    with self.assertRaises(ValueError):
      quantize_provider.set_quantize_activations(
          layer, [quantize_activation, quantize_activation])

  def testGetsResultQuantizers_ReturnsQuantizer(self):
    layer = self._simple_dense_layer()
    quantize_provider = tflite_quantize_registry.TFLiteQuantizeProvider(
        [], [], True)

    output_quantizers = quantize_provider.get_output_quantizers(layer)

    self.assertLen(output_quantizers, 1)
    self._assert_activation_quantizers(output_quantizers)

  def testGetsResultQuantizers_EmptyWhenFalse(self):
    layer = self._simple_dense_layer()
    quantize_provider = tflite_quantize_registry.TFLiteQuantizeProvider(
        [], [], False)

    output_quantizers = quantize_provider.get_output_quantizers(layer)

    self.assertEqual([], output_quantizers)

  def testSerialization(self):
    quantize_provider = tflite_quantize_registry.TFLiteQuantizeProvider(
        ['kernel'], ['activation'], False)

    expected_config = {
        'class_name': 'TFLiteQuantizeProvider',
        'config': {
            'weight_attrs': ['kernel'],
            'activation_attrs': ['activation'],
            'quantize_output': False
        }
    }
    serialized_quantize_provider = serialize_keras_object(quantize_provider)

    self.assertEqual(expected_config, serialized_quantize_provider)

    quantize_provider_from_config = deserialize_keras_object(
        serialized_quantize_provider,
        module_objects=globals(),
        custom_objects=tflite_quantize_registry._types_dict())

    self.assertEqual(quantize_provider, quantize_provider_from_config)


class TFLiteQuantizeProviderRNNTest(test.TestCase, _TestHelper):

  def setUp(self):
    super(TFLiteQuantizeProviderRNNTest, self).setUp()

    self.cell1 = l.LSTMCell(3)
    self.cell2 = l.GRUCell(2)
    self.layer = l.RNN([self.cell1, self.cell2])
    self.layer.build(input_shape=(3, 2))

    self.quantize_provider = tflite_quantize_registry.TFLiteQuantizeProviderRNN(
        [['kernel', 'recurrent_kernel'], ['kernel', 'recurrent_kernel']],
        [['activation', 'recurrent_activation'],
         ['activation', 'recurrent_activation']],
        False
    )

  def _expected_weights(self):
    return [self.cell1.kernel, self.cell1.recurrent_kernel,
            self.cell2.kernel, self.cell2.recurrent_kernel]

  def _expected_activations(self):
    return [self.cell1.activation, self.cell1.recurrent_activation,
            self.cell2.activation, self.cell2.recurrent_activation]

  def _dummy_weights(self, weight):
    return K.variable(np.ones(weight.shape.as_list()))

  def testGetsQuantizeWeightsAndQuantizers(self):
    (weights, weight_quantizers) = self._convert_list(
        self.quantize_provider.get_weights_and_quantizers(self.layer))

    self._assert_weight_quantizers(weight_quantizers)
    self.assertEqual(self._expected_weights(), weights)

  def testGetsQuantizeActivationsAndQuantizers(self):
    (activations, activation_quantizers) = self._convert_list(
        self.quantize_provider.get_activations_and_quantizers(self.layer))

    self._assert_activation_quantizers(activation_quantizers)
    self.assertEqual(self._expected_activations(), activations)

  def testSetsQuantizeWeights(self):
    quantize_weights = [
        self._dummy_weights(self.cell1.kernel),
        self._dummy_weights(self.cell1.recurrent_kernel),
        self._dummy_weights(self.cell2.kernel),
        self._dummy_weights(self.cell2.recurrent_kernel)
    ]

    self.quantize_provider.set_quantize_weights(self.layer, quantize_weights)

    self.assertEqual(self._expected_weights(), quantize_weights)

  def testSetsQuantizeActivations(self):
    quantize_activations = [keras.activations.relu, keras.activations.softmax,
                            keras.activations.relu, keras.activations.softmax]

    self.quantize_provider.set_quantize_activations(
        self.layer, quantize_activations)

    self.assertEqual(self._expected_activations(), quantize_activations)

  def testSetsQuantizeWeights_ErrorOnWrongNumberOfWeights(self):
    with self.assertRaises(ValueError):
      self.quantize_provider.set_quantize_weights(self.layer, [])

    quantize_weights = [
        self._dummy_weights(self.cell1.kernel),
        self._dummy_weights(self.cell1.recurrent_kernel),
    ]
    with self.assertRaises(ValueError):
      self.quantize_provider.set_quantize_weights(
          self.layer, quantize_weights)

  def testSetsQuantizeWeights_ErrorOnWrongShapeOfWeight(self):
    quantize_weights = [
        self._dummy_weights(self.cell1.kernel),
        self._dummy_weights(self.cell1.recurrent_kernel),
        K.variable(np.ones([1, 2])),  # Incorrect shape.
        self._dummy_weights(self.cell2.recurrent_kernel)
    ]

    with self.assertRaises(ValueError):
      self.quantize_provider.set_quantize_weights(self.layer, quantize_weights)

  def testSetsQuantizeActivations_ErrorOnWrongNumberOfActivations(self):
    quantize_activation = keras.activations.relu

    with self.assertRaises(ValueError):
      self.quantize_provider.set_quantize_activations(self.layer, [])

    with self.assertRaises(ValueError):
      self.quantize_provider.set_quantize_activations(
          self.layer, [quantize_activation, quantize_activation])

  def testSerialization(self):
    expected_config = {
        'class_name': 'TFLiteQuantizeProviderRNN',
        'config': {
            'weight_attrs': [['kernel', 'recurrent_kernel'],
                             ['kernel', 'recurrent_kernel']],
            'activation_attrs': [['activation', 'recurrent_activation'],
                                 ['activation', 'recurrent_activation']],
            'quantize_output': False
        }
    }
    serialized_quantize_provider = serialize_keras_object(
        self.quantize_provider)

    self.assertEqual(expected_config, serialized_quantize_provider)

    quantize_provider_from_config = deserialize_keras_object(
        serialized_quantize_provider,
        module_objects=globals(),
        custom_objects=tflite_quantize_registry._types_dict())

    self.assertEqual(self.quantize_provider, quantize_provider_from_config)


class ActivationQuantizeProviderTest(test.TestCase):

  def testRaisesErrorUnsupportedActivation(self):
    quantize_provider = tflite_quantize_registry.ActivationQuantizeProvider()

    with self.assertRaises(ValueError):
      quantize_provider.get_output_quantizers(keras.layers.Activation('tanh'))

    with self.assertRaises(ValueError):
      quantize_provider.get_output_quantizers(
          keras.layers.Activation(lambda x: x))


if __name__ == '__main__':
  test.main()
