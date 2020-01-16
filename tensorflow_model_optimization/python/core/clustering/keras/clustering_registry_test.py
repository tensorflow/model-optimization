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
"""Tests for keras clustering registry API."""

import numpy as np

from tensorflow_model_optimization.python.core.clustering.keras import clusterable_layer
from tensorflow_model_optimization.python.core.clustering.keras import clustering_registry

import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K

from absl.testing import parameterized

keras = tf.keras
errors_impl = tf.errors
layers = keras.layers
test = tf.test

layers = keras.layers
ClusterRegistry = clustering_registry.ClusteringRegistry
ClusteringLookupRegistry = clustering_registry.ClusteringLookupRegistry


class ClusteringAlgorithmTest(parameterized.TestCase):

  def _pull_values(self, ca, pulling_indices, expected_output):
    pulling_indices_np = np.array(pulling_indices)
    res_tf = ca.get_clustered_weight(pulling_indices_np)

    res_np = K.batch_get_value([res_tf])[0]
    res_np_list = res_np.tolist()

    self.assertSequenceEqual(res_np_list, expected_output)

  @parameterized.parameters(
      ([-1, 1], [[0, 0, 1], [1, 1, 1]], [[-1, -1, 1], [1, 1, 1]]),
      ([-1, 0, 1], [[1, 1, 1], [1, 1, 1]], [[0, 0, 0], [0, 0, 0]]),
  )
  def testDenseWeightsCA(self,
                         clustering_centroids,
                         pulling_indices,
                         expected_output):
    ca = clustering_registry.DenseWeightsCA(clustering_centroids)
    self._pull_values(ca, pulling_indices, expected_output)

  @parameterized.parameters(
      ([-1, 1], [0, 0, 0, 0, 1], [-1, -1, -1, -1, 1]),
      ([0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3]),
  )
  def testBiasWeightsCA(self,
                        clustering_centroids,
                        pulling_indices,
                        expected_output):
    ca = clustering_registry.BiasWeightsCA(clustering_centroids)
    self._pull_values(ca, pulling_indices, expected_output)

  @parameterized.parameters(
      ([0, 3], [[[[0, 0, 0], [1, 1, 1], [0, 0, 0]]]],
       [[[[0, 0, 0], [3, 3, 3], [0, 0, 0]]]]),
      ([0, 3, 5], [[[[0, 1, 2], [1, 1, 1], [2, 1, 0]]]],
       [[[[0, 3, 5], [3, 3, 3], [5, 3, 0]]]]),
  )
  def testConvolutionalWeightsCA(self,
                                 clustering_centroids,
                                 pulling_indices,
                                 expected_output):
    ca = clustering_registry.ConvolutionalWeightsCA(clustering_centroids)
    self._pull_values(ca, pulling_indices, expected_output)


class CustomLayer(layers.Layer):
  pass


class ClusteringLookupRegistryTest(test.TestCase, parameterized.TestCase):

  def testLookupHasEverythingFromRegistry(self):
    # So basically we want to make sure that every layer that has non-empty
    # ClusteringRegistry records is also presented in the ClusteringLookup
    for layer, clustering_record in ClusterRegistry._LAYERS_WEIGHTS_MAP.items():
      if clustering_record == []:
        continue

      self.assertIn(layer, ClusteringLookupRegistry._LAYERS_RESHAPE_MAP)

      for cr in clustering_record:
        self.assertIn(cr, ClusteringLookupRegistry._LAYERS_RESHAPE_MAP[layer])

  def testGetClusteringImplFailsWithUnknonwClassUnknownWeight(self):
    with self.assertRaises(ValueError):
      ClusteringLookupRegistry.get_clustering_impl(CustomLayer(),
                                                   'no_such_weight')

  def testGetClusteringImplFailsWithKnonwClassUnknownWeight(self):
    with self.assertRaises(ValueError):
      ClusteringLookupRegistry.get_clustering_impl(layers.Dense(10),
                                                   'no_such_weight')

  @parameterized.parameters(
      (layers.Conv2D, 'kernel', clustering_registry.ConvolutionalWeightsCA),
      (layers.Conv1D, 'kernel', clustering_registry.ConvolutionalWeightsCA),
  )
  def testReturnsResultsForKnownTypeKnownWeights(self,
                                                 layer_type,
                                                 weight,
                                                 expected):
    # layer_type is a class, thus constructing an object here
    self.assertTrue(ClusteringLookupRegistry.get_clustering_impl(
        layer_type(32, 3), weight) is expected)

  def testRegisterNewImplWorks(self):
    class NewKernelCA(clustering_registry.AbstractClusteringAlgorithm):

      def get_pulling_indices(self, weight):
        return 1, 2, 3

    new_impl = {
        CustomLayer: {
            'new_kernel': NewKernelCA
        }
    }

    ClusteringLookupRegistry.register_new_implementation(new_impl)
    self.assertTrue(ClusteringLookupRegistry.get_clustering_impl(
        CustomLayer(), 'new_kernel') is NewKernelCA)

  def testFailsIfNotADictIsGivenAsInput(self):
    with self.assertRaises(TypeError):
      ClusteringLookupRegistry.register_new_implementation([1, 2, 3, 4])

  def testFailsIfNotADictIsGivenAsConcreteImplementation(self):
    with self.assertRaises(TypeError):
      ClusteringLookupRegistry.register_new_implementation({
          ClusteringLookupRegistry: [('new_kernel', lambda x: x)]
      })


class ClusterRegistryTest(test.TestCase):
  class CustomLayerFromClusterableLayer(layers.Dense):
    pass

  class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
      self.units = units
      self.state_size = units
      super(ClusterRegistryTest.MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
      self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='uniform',
                                    name='kernel')
      self.recurrent_kernel = self.add_weight(
          shape=(self.units, self.units),
          initializer='uniform',
          name='recurrent_kernel')
      self.built = True

    def call(self, inputs, states):
      prev_output = states[0]
      h = K.dot(inputs, self.kernel)
      output = h + K.dot(prev_output, self.recurrent_kernel)
      return output, [output]

  class MinimalRNNCellClusterable(MinimalRNNCell,
                                  clusterable_layer.ClusterableLayer):

    def get_clusterable_weights(self):
      return [self.kernel, self.recurrent_kernel]

  def testSupportsKerasClusterableLayer(self):
    self.assertTrue(ClusterRegistry.supports(layers.Dense(10)))

  def testSupportsKerasClusterableLayerAlias(self):
    # layers.Conv2D maps to layers.convolutional.Conv2D
    self.assertTrue(ClusterRegistry.supports(layers.Conv2D(10, 5)))

  def testSupportsKerasNonClusterableLayer(self):
    # Dropout is a layer known to not be clusterable.
    self.assertTrue(ClusterRegistry.supports(layers.Dropout(0.5)))

  def testDoesNotSupportKerasUnsupportedLayer(self):
    # ConvLSTM2D is a built-in keras layer but not supported.
    self.assertFalse(ClusterRegistry.supports(layers.ConvLSTM2D(2, (5, 5))))

  def testSupportsKerasRNNLayers(self):
    self.assertTrue(ClusterRegistry.supports(layers.LSTM(10)))
    self.assertTrue(ClusterRegistry.supports(layers.GRU(10)))
    self.assertTrue(ClusterRegistry.supports(layers.SimpleRNN(10)))

  def testDoesNotSupportKerasRNNLayerUnknownCell(self):
    self.assertFalse(ClusterRegistry.supports(
        keras.layers.RNN(ClusterRegistryTest.MinimalRNNCell(32))))

  def testSupportsKerasRNNLayerClusterableCell(self):
    self.assertTrue(ClusterRegistry.supports(
        keras.layers.RNN(ClusterRegistryTest.MinimalRNNCellClusterable(32))))

  def testDoesNotSupportCustomLayer(self):
    self.assertFalse(ClusterRegistry.supports(CustomLayer(10)))

  def testDoesNotSupportCustomLayerInheritedFromClusterableLayer(self):
    self.assertFalse(
        ClusterRegistry.supports(
            ClusterRegistryTest.CustomLayerFromClusterableLayer(10)))

  def testMakeClusterableRaisesErrorForKerasUnsupportedLayer(self):
    with self.assertRaises(ValueError):
      ClusterRegistry.make_clusterable(layers.ConvLSTM2D(2, (5, 5)))

  def testMakeClusterableRaisesErrorForCustomLayer(self):
    with self.assertRaises(ValueError):
      ClusterRegistry.make_clusterable(CustomLayer(10))

  def testMakeClusterableRaisesErrorForCustomLayerInheritedFromClusterableLayer(
      self):
    with self.assertRaises(ValueError):
      ClusterRegistry.make_clusterable(
          ClusterRegistryTest.CustomLayerFromClusterableLayer(10))

  def testMakeClusterableWorksOnKerasClusterableLayer(self):
    layer = layers.Dense(10)
    with self.assertRaises(AttributeError):
      layer.get_clusterable_weights()

    ClusterRegistry.make_clusterable(layer)
    # Required since build method sets up the layer weights.
    keras.Sequential([layer]).build(input_shape=(10, 1))

    self.assertEqual([('kernel', layer.kernel)],
                     layer.get_clusterable_weights())

  def testMakeClusterableWorksOnKerasNonClusterableLayer(self):
    layer = layers.Dropout(0.5)
    with self.assertRaises(AttributeError):
      layer.get_clusterable_weights()

    ClusterRegistry.make_clusterable(layer)

    self.assertEqual([], layer.get_clusterable_weights())

  def testMakeClusterableWorksOnKerasRNNLayer(self):
    layer = layers.LSTM(10)
    with self.assertRaises(AttributeError):
      layer.get_clusterable_weights()

    ClusterRegistry.make_clusterable(layer)
    keras.Sequential([layer]).build(input_shape=(2, 3, 4))

    self.assertEqual(
        [layer.cell.kernel, layer.cell.recurrent_kernel],
        layer.get_clusterable_weights())

  def testMakeClusterableWorksOnKerasRNNLayerWithRNNCellsParams(self):
    cell1 = layers.LSTMCell(10)
    cell2 = layers.GRUCell(5)
    layer = layers.RNN([cell1, cell2])
    with self.assertRaises(AttributeError):
      layer.get_clusterable_weights()

    ClusterRegistry.make_clusterable(layer)
    keras.Sequential([layer]).build(input_shape=(2, 3, 4))

    expected_weights = [
        cell1.kernel, cell1.recurrent_kernel, cell2.kernel,
        cell2.recurrent_kernel
    ]
    self.assertEqual(expected_weights, layer.get_clusterable_weights())

  def testMakeClusterableWorksOnKerasRNNLayerWithClusterableCell(self):
    cell1 = layers.LSTMCell(10)
    cell2 = ClusterRegistryTest.MinimalRNNCellClusterable(5)
    layer = layers.RNN([cell1, cell2])
    with self.assertRaises(AttributeError):
      layer.get_clusterable_weights()

    ClusterRegistry.make_clusterable(layer)
    keras.Sequential([layer]).build(input_shape=(2, 3, 4))

    expected_weights = [
        cell1.kernel, cell1.recurrent_kernel, cell2.kernel,
        cell2.recurrent_kernel
    ]
    self.assertEqual(expected_weights, layer.get_clusterable_weights())

  def testMakeClusterableRaisesErrorOnRNNLayersUnsupportedCell(self):
    with self.assertRaises(ValueError):
      ClusterRegistry.make_clusterable(layers.RNN(
          [layers.LSTMCell(10), ClusterRegistryTest.MinimalRNNCell(5)]))


if __name__ == '__main__':
  test.main()
