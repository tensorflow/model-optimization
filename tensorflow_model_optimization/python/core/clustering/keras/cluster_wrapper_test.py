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
import numpy as np

from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper
from tensorflow_model_optimization.python.core.clustering.keras import clusterable_layer
from tensorflow_model_optimization.python.core.clustering.keras import clustering_registry

from tensorflow.python.framework import test_util as tf_test_util

import tensorflow.compat.v1 as tf
from absl.testing import parameterized

keras = tf.keras
errors_impl = tf.errors
layers = keras.layers
test = tf.test

layers = keras.layers
ClusterRegistry = clustering_registry.ClusteringRegistry
ClusteringLookupRegistry = clustering_registry.ClusteringLookupRegistry


class NonClusterableLayer(layers.Dense):
  pass


class AlreadyClusterableLayer(layers.Dense, clusterable_layer.ClusterableLayer):

  def get_clusterable_weights(self):
    pass


class ClusterWeightsTest(test.TestCase, parameterized.TestCase):

  def testCannotBeInitializedWithNonLayerObject(self):
    with self.assertRaises(ValueError):
      cluster_wrapper.ClusterWeights({
          'this': 'is not a Layer instance'
      }, number_of_clusters=13, cluster_centroids_init='linear')

  def testCannotBeInitializedWithNonClusterableLayer(self):
    with self.assertRaises(ValueError):
      cluster_wrapper.ClusterWeights(NonClusterableLayer(10),
                                     number_of_clusters=13,
                                     cluster_centroids_init='linear')

  def testCanBeInitializedWithClusterableLayer(self):
    l = cluster_wrapper.ClusterWeights(layers.Dense(10),
                                       number_of_clusters=13,
                                       cluster_centroids_init='linear')
    self.assertIsInstance(l, cluster_wrapper.ClusterWeights)

  def testCannotBeInitializedWithNonIntegerNumberOfClusters(self):
    with self.assertRaises(ValueError):
      cluster_wrapper.ClusterWeights(layers.Dense(10),
                                     number_of_clusters="13",
                                     cluster_centroids_init='linear')

  def testCannotBeInitializedWithFloatNumberOfClusters(self):
    with self.assertRaises(ValueError):
      cluster_wrapper.ClusterWeights(layers.Dense(10),
                                     number_of_clusters=13.4,
                                     cluster_centroids_init='linear')

  @parameterized.parameters(
      (0),
      (1),
      (-32)
  )
  def testCannotBeInitializedWithNumberOfClustersLessThanTwo(
      self, number_of_clusters):
    with self.assertRaises(ValueError):
      cluster_wrapper.ClusterWeights(layers.Dense(10),
                                     number_of_clusters=number_of_clusters,
                                     cluster_centroids_init='linear')

  def testCanBeInitializedWithAlreadyClusterableLayer(self):
    layer = AlreadyClusterableLayer(10)
    l = cluster_wrapper.ClusterWeights(layer,
                                       number_of_clusters=13,
                                       cluster_centroids_init='linear')
    self.assertIsInstance(l, cluster_wrapper.ClusterWeights)

  def testIfLayerHasBatchShapeClusterWeightsMustHaveIt(self):
    l = cluster_wrapper.ClusterWeights(layers.Dense(10, input_shape=(10,)),
                                       number_of_clusters=13,
                                       cluster_centroids_init='linear')
    self.assertTrue(hasattr(l, '_batch_input_shape'))

  # Makes it easier to test all possible parameters combinations.
  @parameterized.parameters(
      *itertools.product(range(2, 16, 4), ('linear', 'random', 'density-based'))
  )
  def testValuesAreClusteredAfterStripping(self,
                                           number_of_clusters,
                                           cluster_centroids_init):
    # We want to make sure that for any number of clusters and any
    # initializations methods there is always no more than number_of_clusters
    # unique points after stripping the model
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


if __name__ == '__main__':
  tf.disable_v2_behavior()
  test.main()
