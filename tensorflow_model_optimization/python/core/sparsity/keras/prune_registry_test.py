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
"""Tests for prune registry."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry

keras = tf.keras
layers = keras.layers
PruneRegistry = prune_registry.PruneRegistry


class CustomLayer(layers.Layer):
  pass


class CustomLayerFromPrunableLayer(layers.Dense):
  pass


class MinimalRNNCell(keras.layers.Layer):

  def __init__(self, units, **kwargs):
    self.units = units
    self.state_size = units
    super(MinimalRNNCell, self).__init__(**kwargs)

  def build(self, input_shape):
    self.kernel = self.add_weight(
        shape=(input_shape[-1], self.units),
        initializer='uniform',
        name='kernel')
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units),
        initializer='uniform',
        name='recurrent_kernel')
    self.built = True

  def call(self, inputs, states):
    prev_output = states[0]
    h = keras.backend.dot(inputs, self.kernel)
    output = h + keras.backend.dot(prev_output, self.recurrent_kernel)
    return output, [output]


class MinimalRNNCellPrunable(MinimalRNNCell, prunable_layer.PrunableLayer):

  def get_prunable_weights(self):
    return [self.kernel, self.recurrent_kernel]


class PruneRegistryTest(tf.test.TestCase, parameterized.TestCase):

  _PRUNE_REGISTRY_SUPPORTED_LAYERS = [
      # Supports basic Keras layers even though it is not prunbale.
      layers.Dense(10),
      layers.Conv2D(10, 5),
      layers.Dropout(0.5),
      # Supports specific layers from experimental or compat_v1.
      layers.experimental.SyncBatchNormalization(),
      layers.experimental.preprocessing.Rescaling,
      tf.compat.v1.keras.layers.BatchNormalization(),
      # Supports Keras RNN Layers with prunable cells.
      layers.LSTM(10),
      layers.GRU(10),
      layers.SimpleRNN(10),
      layers.RNN(layers.LSTMCell(10)),
      layers.RNN([
          layers.LSTMCell(10),
          layers.GRUCell(10),
          keras.experimental.PeepholeLSTMCell(10),
          layers.SimpleRNNCell(10)
      ]),
      keras.layers.RNN(MinimalRNNCellPrunable(32)),
  ]

  @parameterized.parameters(_PRUNE_REGISTRY_SUPPORTED_LAYERS)
  def testSupportsLayer(self, layer):
    self.assertTrue(PruneRegistry.supports(layer))

  _PRUNE_REGISTRY_UNSUPPORTED_LAYERS = [
      # Not support a few built-in keras layers.
      layers.ConvLSTM2D(2, (5, 5)),
      # Not support RNN layers with unknown cell
      keras.layers.RNN(MinimalRNNCell(32)),
      # Not support Custom layers, even though inherited from prunable layer.
      CustomLayer(),
      CustomLayerFromPrunableLayer(10),
  ]

  @parameterized.parameters(_PRUNE_REGISTRY_UNSUPPORTED_LAYERS)
  def testDoesNotSupportLayer(self, layer):
    self.assertFalse(PruneRegistry.supports(layer))

  def testMakePrunableRaisesErrorForKerasUnsupportedLayer(self):
    with self.assertRaises(ValueError):
      PruneRegistry.make_prunable(layers.ConvLSTM2D(2, (5, 5)))

  def testMakePrunableRaisesErrorForCustomLayer(self):
    with self.assertRaises(ValueError):
      PruneRegistry.make_prunable(CustomLayer())

  def testMakePrunableRaisesErrorForCustomLayerInheritedFromPrunableLayer(self):
    with self.assertRaises(ValueError):
      PruneRegistry.make_prunable(CustomLayerFromPrunableLayer(10))

  def testMakePrunableWorksOnKerasPrunableLayer(self):
    layer = layers.Dense(10)
    with self.assertRaises(AttributeError):
      layer.get_prunable_weights()

    PruneRegistry.make_prunable(layer)
    # Required since build method sets up the layer weights.
    keras.Sequential([layer]).build(input_shape=(10, 1))

    self.assertEqual([layer.kernel], layer.get_prunable_weights())

  def testMakePrunableWorksOnKerasNonPrunableLayer(self):
    layer = layers.Dropout(0.5)
    with self.assertRaises(AttributeError):
      layer.get_prunable_weights()

    PruneRegistry.make_prunable(layer)

    self.assertEqual([], layer.get_prunable_weights())

  def testMakePrunableWorksOnKerasRNNLayer(self):
    layer = layers.LSTM(10)
    with self.assertRaises(AttributeError):
      layer.get_prunable_weights()

    PruneRegistry.make_prunable(layer)
    keras.Sequential([layer]).build(input_shape=(2, 3, 4))

    self.assertEqual(
        [layer.cell.kernel, layer.cell.recurrent_kernel],
        layer.get_prunable_weights())

  def testMakePrunableWorksOnKerasRNNLayerWithRNNCellsParams(self):
    cell1 = layers.LSTMCell(10)
    cell2 = layers.GRUCell(5)
    layer = layers.RNN([cell1, cell2])
    with self.assertRaises(AttributeError):
      layer.get_prunable_weights()

    PruneRegistry.make_prunable(layer)
    keras.Sequential([layer]).build(input_shape=(2, 3, 4))

    expected_weights = [
        cell1.kernel, cell1.recurrent_kernel, cell2.kernel,
        cell2.recurrent_kernel
    ]
    self.assertEqual(expected_weights, layer.get_prunable_weights())

  def testMakePrunableWorksOnKerasRNNLayerWithPrunableCell(self):
    cell1 = layers.LSTMCell(10)
    cell2 = MinimalRNNCellPrunable(5)
    layer = layers.RNN([cell1, cell2])
    with self.assertRaises(AttributeError):
      layer.get_prunable_weights()

    PruneRegistry.make_prunable(layer)
    keras.Sequential([layer]).build(input_shape=(2, 3, 4))

    expected_weights = [
        cell1.kernel, cell1.recurrent_kernel, cell2.kernel,
        cell2.recurrent_kernel
    ]
    self.assertEqual(expected_weights, layer.get_prunable_weights())

  def testMakePrunableRaisesErrorOnRNNLayersUnsupportedCell(self):
    with self.assertRaises(ValueError):
      PruneRegistry.make_prunable(
          layers.RNN([layers.LSTMCell(10),
                      MinimalRNNCell(5)]))


if __name__ == '__main__':
  tf.test.main()
