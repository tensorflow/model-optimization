# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Distributed clustering test."""

import unittest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.keras import test_utils as keras_test_utils
from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper

keras = tf.keras
CentroidInitialization = cluster_config.CentroidInitialization


def _distribution_strategies():
  return [
      tf.distribute.MirroredStrategy()
  ]


class ClusterDistributedTest(tf.test.TestCase, parameterized.TestCase):
  """Distributed tests for clustering."""

  def setUp(self):
    super(ClusterDistributedTest, self).setUp()
    self.params = {
        "number_of_clusters": 2,
        "cluster_centroids_init": CentroidInitialization.LINEAR
    }

  @unittest.skip("MirroredVariable doesn't works with tf.custom_gradient.")
  @parameterized.parameters(_distribution_strategies())
  def testClusterSimpleDenseModel(self, distribution):
    """End-to-end test."""
    with distribution.scope():
      model = cluster.cluster_weights(
          keras_test_utils.build_simple_dense_model(), **self.params)
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])

    model.summary()
    model.fit(
        np.random.rand(20, 10),
        keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        epochs=1,
        batch_size=20)
    model.predict(np.random.rand(20, 10))

    stripped_model = cluster.strip_clustering(model)
    weights_as_list = stripped_model.layers[0].kernel.numpy().reshape(-1,).tolist()
    unique_weights = set(weights_as_list)
    self.assertLessEqual(len(unique_weights), self.params["number_of_clusters"])

  @unittest.skip("MirroredVariable doesn't works with tf.custom_gradient.")
  @parameterized.parameters(_distribution_strategies())
  def testAssociationValuesPerReplica(self, distribution):
    """Verifies that associations of weights are updated per replica."""
    assert tf.distribute.get_replica_context() is not None
    with distribution.scope():
      assert tf.distribute.get_replica_context() is None
      input_shape = (1, 2)
      output_shape = (2, 8)
      l = cluster_wrapper.ClusterWeights(
          keras.layers.Dense(8, input_shape=input_shape),
          number_of_clusters=self.params["number_of_clusters"],
          cluster_centroids_init=self.params["cluster_centroids_init"]
      )
      l.build(input_shape)

      clusterable_weights = l.layer.get_clusterable_weights()
      self.assertEqual(len(clusterable_weights), 1)
      weights_name = clusterable_weights[0][0]
      self.assertEqual(weights_name, 'kernel')
      centroids1 = l.cluster_centroids[weights_name]

      mean_weight = tf.reduce_mean(l.layer.kernel)
      min_weight = tf.reduce_min(l.layer.kernel)
      max_weight = tf.reduce_max(l.layer.kernel)
      max_dist = max_weight - min_weight

      def assert_all_cluster_indices(per_replica, indices_val):
        if indices_val == 1:
          val_tensor = tf.dtypes.cast(
              tf.ones(shape=output_shape), per_replica[0].dtype)
        if indices_val == 0:
          val_tensor = tf.dtypes.cast(
              tf.zeros(shape=output_shape), per_replica[0].dtype)
        for i in range(0, len(per_replica)):
          all_equal = tf.reduce_all(
              tf.equal(
                  per_replica[i], val_tensor
              )
          )
          self.assertTrue(all_equal)

      def update_fn(v, val):
        return v.assign(val)

      initial_val = tf.Variable([mean_weight, mean_weight + 2.0 * max_dist], \
        aggregation=tf.VariableAggregation.MEAN)

      centroids1 = distribution.extended.update(
          centroids1, update_fn, args=(initial_val,))
      l.call(tf.ones(shape=input_shape))

      clst_indices = l.pulling_indices[weights_name]
      per_replica = distribution.experimental_local_results(clst_indices)
      assert_all_cluster_indices(per_replica, 0)

      second_val = tf.Variable([mean_weight - 2.0 * max_dist, mean_weight], \
        aggregation=tf.VariableAggregation.MEAN)
      centroids2 = l.cluster_centroids[weights_name]
      centroids2 = distribution.extended.update(
          centroids2, update_fn, args=(second_val,))
      l.call(tf.ones(shape=input_shape))

      clst_indices = l.pulling_indices[weights_name]
      per_replica = distribution.experimental_local_results(clst_indices)
      assert_all_cluster_indices(per_replica, 1)

if __name__ == '__main__':
  tf.test.main()
