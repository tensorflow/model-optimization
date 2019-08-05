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

from tensorflow.python import keras
from tensorflow.python.keras import layers as l
from tensorflow.python.platform import test

from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.tflite import tflite_quantize_registry


class TFLiteQuantizeRegistryTest(test.TestCase):

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

  # TODO(pulkitb): Consider asserting on full equality for quantizers.

  def _assert_weight_quantizers(self, quantizer_list):
    for quantizer in quantizer_list:
      self.assertIsInstance(quantizer, quantizers.LastValueQuantizer)

  def _assert_activation_quantizers(self, quantizer_list):
    for quantizer in quantizer_list:
      self.assertIsInstance(quantizer, quantizers.MovingAverageQuantizer)

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


if __name__ == '__main__':
  test.main()
