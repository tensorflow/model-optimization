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
"""Tests for keras clustering API."""

import json

import tensorflow.compat.v1 as tf
from absl.testing import parameterized

keras = tf.keras
errors_impl = tf.errors
layers = keras.layers
test = tf.test

from tensorflow_model_optimization.python.core.clustering.keras import clusterable_layer
from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import clustering_registry
from tensorflow_model_optimization.python.core.clustering.keras.cluster_wrapper import ClusterWeights

from tensorflow.python.framework import test_util as tf_test_util

class TestModel(keras.Model):
  """A model subclass."""

  def __init__(self):
    """A test subclass model with one dense layer."""
    super(TestModel, self).__init__(name='test_model')
    self.layer1 = keras.layers.Dense(10, activation='relu')

  def call(self, inputs):
    return self.layer1(inputs)


class CustomClusterableLayer(layers.Dense, clusterable_layer.ClusterableLayer):

  def get_clusterable_weights(self):
    return [('kernel', self.kernel)]


class CustomNonClusterableLayer(layers.Dense):
  pass


class ClusterTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ClusterTest, self).setUp()

    self.keras_clusterable_layer = layers.Dense(10)
    self.keras_non_clusterable_layer = layers.Dropout(0.4)
    self.keras_unsupported_layer = layers.ConvLSTM2D(2, (5, 5))  # Unsupported
    self.custom_clusterable_layer = CustomClusterableLayer(10)
    self.custom_non_clusterable_layer = CustomNonClusterableLayer(10)

    clustering_registry.ClusteringLookupRegistry.register_new_implementation(
      {
        CustomClusterableLayer: {
          'kernel': clustering_registry.DenseWeightsCA
        }
      }
    )

    self.model = keras.Sequential()
    self.params = {
      'number_of_clusters': 8,
      'cluster_centroids_init': 'density-based'
    }

  def _build_clustered_layer_model(self, layer):
    wrapped_layer = cluster.cluster_weights(layer, **self.params)
    self.model.add(wrapped_layer)
    self.model.build(input_shape=(10, 1))

    return wrapped_layer

  def _validate_clustered_layer(self, original_layer, wrapped_layer):
    self.assertIsInstance(wrapped_layer, ClusterWeights)
    self.assertEqual(original_layer, wrapped_layer.layer)

  @staticmethod
  def _count_clustered_layers(model):
    count = 0
    for layer in model._layers:
      if isinstance(layer, ClusterWeights):
        count += 1
    return count

  @tf_test_util.run_in_graph_and_eager_modes
  def testClusterKerasClusterableLayer(self):
    wrapped_layer = self._build_clustered_layer_model(self.keras_clusterable_layer)

    self._validate_clustered_layer(self.keras_clusterable_layer, wrapped_layer)

  @tf_test_util.run_in_graph_and_eager_modes
  def testClusterKerasNonClusterableLayer(self):
    wrapped_layer = self._build_clustered_layer_model(self.keras_non_clusterable_layer)

    self._validate_clustered_layer(self.keras_non_clusterable_layer, wrapped_layer)
    self.assertEqual([], wrapped_layer.layer.get_clusterable_weights())

  def testClusterKerasUnsupportedLayer(self):
    with self.assertRaises(ValueError):
      cluster.cluster_weights(self.keras_unsupported_layer, **self.params)

  @tf_test_util.run_in_graph_and_eager_modes
  def testClusterCustomClusterableLayer(self):
    wrapped_layer = self._build_clustered_layer_model(self.custom_clusterable_layer)

    self._validate_clustered_layer(self.custom_clusterable_layer, wrapped_layer)
    self.assertEqual([('kernel', wrapped_layer.layer.kernel)], wrapped_layer.layer.get_clusterable_weights())

  def testClusterCustomNonClusterableLayer(self):
    with self.assertRaises(ValueError):
      ClusterWeights(self.custom_non_clusterable_layer, **self.params)

  @tf_test_util.run_in_graph_and_eager_modes
  def testClusterModelValidLayersSuccessful(self):
    model = keras.Sequential([
      self.keras_clusterable_layer,
      self.keras_non_clusterable_layer,
      self.custom_clusterable_layer
    ])
    clustered_model = cluster.cluster_weights(model, **self.params)
    clustered_model.build(input_shape=(1, 28, 28, 1))

    self.assertEqual(len(model.layers), len(clustered_model.layers))
    for layer, clustered_layer in zip(model.layers, clustered_model.layers):
      self._validate_clustered_layer(layer, clustered_layer)

  def testClusterModelUnsupportedKerasLayerRaisesError(self):
    with self.assertRaises(ValueError):
      cluster.cluster_weights(
        keras.Sequential([
          self.keras_clusterable_layer, self.keras_non_clusterable_layer,
          self.custom_clusterable_layer, self.keras_unsupported_layer
        ]), **self.params)

  def testClusterModelCustomNonClusterableLayerRaisesError(self):
    with self.assertRaises(ValueError):
      cluster.cluster_weights(
        keras.Sequential([
          self.keras_clusterable_layer, self.keras_non_clusterable_layer,
          self.custom_clusterable_layer, self.custom_non_clusterable_layer
        ]), **self.params)

  @tf_test_util.run_in_graph_and_eager_modes
  def testClusterModelDoesNotWrapAlreadyWrappedLayer(self):
    model = keras.Sequential(
      [
        layers.Flatten(),
        cluster.cluster_weights(layers.Dense(10), **self.params),
      ])
    clustered_model = cluster.cluster_weights(model, **self.params)
    clustered_model.build(input_shape=(10, 10, 1))

    self.assertEqual(len(model.layers), len(clustered_model.layers))
    self._validate_clustered_layer(model.layers[0], clustered_model.layers[0])
    # Second layer is used as-is since it's already a clustered layer.
    self.assertEqual(model.layers[1], clustered_model.layers[1])
    self._validate_clustered_layer(model.layers[1].layer, clustered_model.layers[1])

  def testClusterValidLayersListSuccessful(self):
    model_layers = [
      self.keras_clusterable_layer,
      self.keras_non_clusterable_layer,
      self.custom_clusterable_layer
    ]
    clustered_list = cluster.cluster_weights(model_layers, **self.params)

    self.assertEqual(len(model_layers), len(clustered_list))
    for layer, clustered_layer in zip(model_layers, clustered_list):
      self._validate_clustered_layer(layer, clustered_layer)

  def testClusterSequentialModelNoInput(self):
    # No InputLayer
    model = keras.Sequential([
      layers.Dense(10),
      layers.Dense(10),
    ])
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(self._count_clustered_layers(clustered_model), 2)

  @tf_test_util.run_in_graph_and_eager_modes
  def testClusterSequentialModelWithInput(self):
    # With InputLayer
    model = keras.Sequential([
      layers.Dense(10, input_shape=(10,)),
      layers.Dense(10),
    ])
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(self._count_clustered_layers(clustered_model), 2)

  def testClusterSequentialModelPreservesBuiltStateNoInput(self):
    # No InputLayer
    model = keras.Sequential([
      layers.Dense(10),
      layers.Dense(10),
    ])
    self.assertEqual(model.built, False)
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(model.built, False)

    # Test built state is preserved across serialization
    with cluster.cluster_scope():
      loaded_model = keras.models.model_from_config(
        json.loads(clustered_model.to_json()))
      self.assertEqual(loaded_model.built, False)

  @tf_test_util.run_in_graph_and_eager_modes
  def testClusterSequentialModelPreservesBuiltStateWithInput(self):
    # With InputLayer
    model = keras.Sequential([
      layers.Dense(10, input_shape=(10,)),
      layers.Dense(10),
    ])
    self.assertEqual(model.built, True)
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(model.built, True)

    # Test built state is preserved across serialization
    with cluster.cluster_scope():
      loaded_model = keras.models.model_from_config(
        json.loads(clustered_model.to_json()))
    self.assertEqual(loaded_model.built, True)

  @tf_test_util.run_in_graph_and_eager_modes
  def testClusterFunctionalModelPreservesBuiltState(self):
    i1 = keras.Input(shape=(10,))
    i2 = keras.Input(shape=(10,))
    x1 = layers.Dense(10)(i1)
    x2 = layers.Dense(10)(i2)
    outputs = layers.Add()([x1, x2])
    model = keras.Model(inputs=[i1, i2], outputs=outputs)
    self.assertEqual(model.built, True)
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(model.built, True)

    # Test built state preserves across serialization
    with cluster.cluster_scope():
      loaded_model = keras.models.model_from_config(
        json.loads(clustered_model.to_json()))
    self.assertEqual(loaded_model.built, True)

  @tf_test_util.run_in_graph_and_eager_modes
  def testClusterFunctionalModel(self):
    i1 = keras.Input(shape=(10,))
    i2 = keras.Input(shape=(10,))
    x1 = layers.Dense(10)(i1)
    x2 = layers.Dense(10)(i2)
    outputs = layers.Add()([x1, x2])
    model = keras.Model(inputs=[i1, i2], outputs=outputs)
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(self._count_clustered_layers(clustered_model), 3)

  @tf_test_util.run_in_graph_and_eager_modes
  def testClusterFunctionalModelWithLayerReused(self):
    # The model reuses the Dense() layer. Make sure it's only clustered once.
    inp = keras.Input(shape=(10,))
    dense_layer = layers.Dense(10)
    x = dense_layer(inp)
    x = dense_layer(x)
    model = keras.Model(inputs=[inp], outputs=[x])
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(self._count_clustered_layers(clustered_model), 1)

  @tf_test_util.run_in_graph_and_eager_modes
  def testClusterSubclassModel(self):
    model = TestModel()
    with self.assertRaises(ValueError):
      _ = cluster.cluster_weights(model, **self.params)

  @tf_test_util.run_in_graph_and_eager_modes
  def testStripClusteringSequentialModel(self):
    model = keras.Sequential([
      layers.Dense(10),
      layers.Dense(10),
    ])

    clustered_model = cluster.cluster_weights(model, **self.params)
    stripped_model = cluster.strip_clustering(clustered_model)
    self.assertEqual(self._count_clustered_layers(stripped_model), 0)
    self.assertEqual(model.get_config(), stripped_model.get_config())

  @tf_test_util.run_in_graph_and_eager_modes
  def testClusterStrippingFunctionalModel(self):
    i1 = keras.Input(shape=(10,))
    i2 = keras.Input(shape=(10,))
    x1 = layers.Dense(10)(i1)
    x2 = layers.Dense(10)(i2)
    outputs = layers.Add()([x1, x2])
    model = keras.Model(inputs=[i1, i2], outputs=outputs)

    clustered_model = cluster.cluster_weights(model, **self.params)
    stripped_model = cluster.strip_clustering(clustered_model)

    self.assertEqual(self._count_clustered_layers(stripped_model), 0)
    self.assertEqual(model.get_config(), stripped_model.get_config())

  @tf_test_util.run_in_graph_and_eager_modes
  def testClusterWeightsStrippedWeights(self):

    i1 = keras.Input(shape=(10,))
    x1 = layers.BatchNormalization()(i1)
    outputs = x1
    model = keras.Model(inputs=[i1], outputs=outputs)

    clustered_model = cluster.cluster_weights(model, **self.params)
    cluster_weight_length = (len(clustered_model.get_weights()))
    stripped_model = cluster.strip_clustering(clustered_model)

    self.assertEqual(self._count_clustered_layers(stripped_model), 0)
    self.assertEqual(len(stripped_model.get_weights()), cluster_weight_length)

if __name__ == '__main__':
  tf.disable_v2_behavior()
  test.main()
