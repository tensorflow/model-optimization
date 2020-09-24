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
"""Tests for keras ClusterWeights wrapper API."""

import itertools
import tensorflow as tf

from absl.testing import parameterized

from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper
from tensorflow_model_optimization.python.core.clustering.keras import clusterable_layer

keras = tf.keras
errors_impl = tf.errors
layers = keras.layers
test = tf.test

CentroidInitialization = cluster_config.CentroidInitialization


class NonClusterableLayer(layers.Layer):
  """"A custom layer with weights that is not clusterable."""
  def __init__(self, units=10):
      super(NonClusterableLayer, self).__init__()
      self.add_weight(shape=(1, units),
                      initializer='uniform',
                      name='kernel')

  def call(self, inputs):
    return tf.matmul(inputs, self.weights)

class AlreadyClusterableLayer(layers.Dense, clusterable_layer.ClusterableLayer):
  """A custom layer that is clusterable."""

  def get_clusterable_weights(self):
    pass


class ClusterWeightsTest(test.TestCase, parameterized.TestCase):
  """Unit tests for the cluster_wrapper module."""

  def testCannotBeInitializedWithNonLayerObject(self):
    """
    Verifies that ClusterWeights cannot be initialized with an object that is
    not an instance of keras.layers.Layer.
    """
    with self.assertRaises(ValueError):
      cluster_wrapper.ClusterWeights(
          {'this': 'is not a Layer instance'},
          number_of_clusters=13,
          cluster_centroids_init=CentroidInitialization.LINEAR
      )

  def testCannotBeInitializedWithNonClusterableLayer(self):
    """
    Verifies that ClusterWeights cannot be initialized with a non-clusterable
    custom layer.
    """
    with self.assertRaises(ValueError):
      cluster_wrapper.ClusterWeights(
          NonClusterableLayer(10),
          number_of_clusters=13,
          cluster_centroids_init=CentroidInitialization.LINEAR
      )

  def testCanBeInitializedWithClusterableLayer(self):
    """
    Verifies that ClusterWeights can be initialized with a built-in clusterable
    layer.
    """
    l = cluster_wrapper.ClusterWeights(
        layers.Dense(10),
        number_of_clusters=13,
        cluster_centroids_init=CentroidInitialization.LINEAR
    )
    self.assertIsInstance(l, cluster_wrapper.ClusterWeights)

  def testCannotBeInitializedWithNonIntegerNumberOfClusters(self):
    """
    Verifies that ClusterWeights cannot be initialized with a string value
    provided for the number of clusters.
    """
    with self.assertRaises(ValueError):
      cluster_wrapper.ClusterWeights(
          layers.Dense(10),
          number_of_clusters="13",
          cluster_centroids_init=CentroidInitialization.LINEAR
      )

  def testCannotBeInitializedWithFloatNumberOfClusters(self):
    """
    Verifies that ClusterWeights cannot be initialized with a decimal value
    provided for the number of clusters.
    """
    with self.assertRaises(ValueError):
      cluster_wrapper.ClusterWeights(
          layers.Dense(10),
          number_of_clusters=13.4,
          cluster_centroids_init=CentroidInitialization.LINEAR
      )

  @parameterized.parameters(
      (0),
      (1),
      (-32)
  )
  def testCannotBeInitializedWithNumberOfClustersLessThanTwo(
      self, number_of_clusters):
    """
    Verifies that ClusterWeights cannot be initialized with less than two
    clusters.
    """
    with self.assertRaises(ValueError):
      cluster_wrapper.ClusterWeights(
          layers.Dense(10),
          number_of_clusters=number_of_clusters,
          cluster_centroids_init=CentroidInitialization.LINEAR
      )

  @parameterized.parameters(
      (0),
      (2),
      (-32)
  )
  def testCannotBeInitializedWithSparsityPreservationAndNumberOfClustersLessThanThree(
      self, number_of_clusters):
    """
    Verifies that ClusterWeights cannot be initialized with less than three
    clusters when sparsity preservation is enabled.
    """
    with self.assertRaises(ValueError):
      cluster_wrapper.ClusterWeights(
          layers.Dense(10),
          number_of_clusters=number_of_clusters,
          cluster_centroids_init=CentroidInitialization.LINEAR,
          preserve_sparsity=True
      )

  def testCanBeInitializedWithAlreadyClusterableLayer(self):
    """
    Verifies that ClusterWeights can be initialized with a custom clusterable
    layer.
    """
    layer = AlreadyClusterableLayer(10)
    l = cluster_wrapper.ClusterWeights(
        layer,
        number_of_clusters=13,
        cluster_centroids_init=CentroidInitialization.LINEAR
    )
    self.assertIsInstance(l, cluster_wrapper.ClusterWeights)

  def testIfLayerHasBatchShapeClusterWeightsMustHaveIt(self):
    """
    Verifies that the ClusterWeights instance created from a layer that has
    a batch shape attribute, will also have this attribute.
    """
    l = cluster_wrapper.ClusterWeights(
        layers.Dense(10, input_shape=(10,)),
        number_of_clusters=13,
        cluster_centroids_init=CentroidInitialization.LINEAR
    )
    self.assertTrue(hasattr(l, '_batch_input_shape'))

  # Makes it easier to test all possible parameters combinations.
  @parameterized.parameters(
      *itertools.product(
          range(2, 16, 4),
          (
              CentroidInitialization.LINEAR,
              CentroidInitialization.RANDOM,
              CentroidInitialization.DENSITY_BASED
          )
      )
  )
  def testValuesAreClusteredAfterStripping(self,
                                           number_of_clusters,
                                           cluster_centroids_init):
    """
    Verifies that, for any number of clusters and any centroid initialization
    method, the number of unique weight values after stripping is always less
    or equal to number_of_clusters.
    """
    original_model = tf.keras.Sequential([
        layers.Dense(32, input_shape=(10,)),
    ])
    clustered_model = cluster.cluster_weights(
        original_model,
        number_of_clusters=number_of_clusters,
        cluster_centroids_init=cluster_centroids_init
    )
    stripped_model = cluster.strip_clustering(clustered_model)
    weights_as_list = stripped_model.get_weights()[0].reshape(-1,).tolist()
    unique_weights = set(weights_as_list)
    # Make sure numbers match
    self.assertLessEqual(len(unique_weights), number_of_clusters)

    # Make sure that the stripped layer is the Dense one
    self.assertIsInstance(stripped_model.layers[0], layers.Dense)

  def testClusterReassociation(self):
    """
    Verifies that the association of weights to cluster centroids are updated
    every iteration.
    """

    # Create a dummy layer for this test
    input_shape = (1, 2,)
    l = cluster_wrapper.ClusterWeights(
        keras.layers.Dense(8, input_shape=input_shape),
        number_of_clusters=2,
        cluster_centroids_init=CentroidInitialization.LINEAR
    )
    # Build a layer with the given shape
    l.build(input_shape)

    # Get name of the clusterable weights
    clusterable_weights = l.layer.get_clusterable_weights()
    self.assertEqual(len(clusterable_weights), 1)
    weights_name = clusterable_weights[0][0]
    self.assertEqual(weights_name, 'kernel')
    # Get cluster centroids
    centroids = l.cluster_centroids_tf[weights_name]

    # Calculate some statistics of the weights to set the centroids later on
    mean_weight = tf.reduce_mean(l.layer.kernel)
    min_weight = tf.reduce_min(l.layer.kernel)
    max_weight = tf.reduce_max(l.layer.kernel)
    max_dist = max_weight - min_weight

    def assert_all_weights_associated(weights, centroid_index):
      """Helper function to make sure that all weights are associated with one
      centroid."""
      all_associated = tf.reduce_all(
          tf.equal(
              weights,
              tf.constant(centroids[centroid_index], shape=weights.shape)
          )
      )
      self.assertTrue(all_associated)

    # Set centroids so that all weights should be re-associated with centroid 0
    centroids[0].assign(mean_weight)
    centroids[1].assign(mean_weight + 2.0 * max_dist)

    # Update associations of weights to centroids
    l.call(tf.ones(shape=input_shape))

    # Weights should now be all clustered with the centroid 0
    assert_all_weights_associated(l.layer.kernel, centroid_index=0)

    # Set centroids so that all weights should be re-associated with centroid 1
    centroids[0].assign(mean_weight - 2.0 * max_dist)
    centroids[1].assign(mean_weight)

    # Update associations of weights to centroids
    l.call(tf.ones(shape=input_shape))

    # Weights should now be all clustered with the centroid 1
    assert_all_weights_associated(l.layer.kernel, centroid_index=1)


  def testClusterReassociation(self):
    """
    Verifies that the association of weights to cluster centroids are updated
    every iteration.
    """

    # Create a dummy layer for this test
    input_shape = (1, 2,)
    l = cluster_wrapper.ClusterWeights(
        keras.layers.Dense(8, input_shape=input_shape),
        number_of_clusters=2,
        cluster_centroids_init=CentroidInitialization.LINEAR
    )
    # Build a layer with the given shape
    l.build(input_shape)

    # Get name of the clusterable weights
    clusterable_weights = l.layer.get_clusterable_weights()
    self.assertEqual(len(clusterable_weights), 1)
    weights_name = clusterable_weights[0][0]
    self.assertEqual(weights_name, 'kernel')
    # Get cluster centroids
    centroids = l.cluster_centroids_tf[weights_name]

    # Calculate some statistics of the weights to set the centroids later on
    mean_weight = tf.reduce_mean(l.layer.kernel)
    min_weight = tf.reduce_min(l.layer.kernel)
    max_weight = tf.reduce_max(l.layer.kernel)
    max_dist = max_weight - min_weight

    def assert_all_weights_associated(weights, centroid_index):
      """Helper function to make sure that all weights are associated with one
      centroid."""
      all_associated = tf.reduce_all(
          tf.equal(
              weights,
              tf.constant(centroids[centroid_index], shape=weights.shape)
          )
      )
      self.assertTrue(all_associated)

    # Set centroids so that all weights should be re-associated with centroid 0
    centroids[0].assign(mean_weight)
    centroids[1].assign(mean_weight + 2.0 * max_dist)

    # Update associations of weights to centroids
    l.call(tf.ones(shape=input_shape))

    # Weights should now be all clustered with the centroid 0
    assert_all_weights_associated(l.layer.kernel, centroid_index=0)

    # Set centroids so that all weights should be re-associated with centroid 1
    centroids[0].assign(mean_weight - 2.0 * max_dist)
    centroids[1].assign(mean_weight)

    # Update associations of weights to centroids
    l.call(tf.ones(shape=input_shape))

    # Weights should now be all clustered with the centroid 1
    assert_all_weights_associated(l.layer.kernel, centroid_index=1)


if __name__ == '__main__':
  test.main()
