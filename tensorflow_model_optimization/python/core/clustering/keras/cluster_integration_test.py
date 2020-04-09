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
"""End-to-end tests for keras clustering API."""

import numpy as np
import tensorflow as tf

from absl.testing import parameterized
from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.clustering.keras import cluster


keras = tf.keras
layers = keras.layers
test = tf.test


class ClusterIntegrationTest(test.TestCase, parameterized.TestCase):
  """Integration tests for clustering."""

  @keras_parameterized.run_all_keras_modes
  def testValuesRemainClusteredAfterTraining(self):
    """
    Verifies that training a clustered model does not destroy the clusters.
    """
    number_of_clusters = 10
    original_model = keras.Sequential([
        layers.Dense(2, input_shape=(2,)),
        layers.Dense(2),
    ])

    clustered_model = cluster.cluster_weights(
        original_model,
        number_of_clusters=number_of_clusters,
        cluster_centroids_init='linear'
    )

    clustered_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy']
    )

    def dataset_generator():
      x_train = np.array([
          [0, 1],
          [2, 0],
          [0, 3],
          [4, 1],
          [5, 1],
      ])
      y_train = np.array([
          [0, 1],
          [1, 0],
          [1, 0],
          [0, 1],
          [0, 1],
      ])
      for x, y in zip(x_train, y_train):
        yield np.array([x]), np.array([y])

    clustered_model.fit_generator(dataset_generator(), steps_per_epoch=1)
    stripped_model = cluster.strip_clustering(clustered_model)
    weights_as_list = stripped_model.get_weights()[0].reshape(-1,).tolist()
    unique_weights = set(weights_as_list)
    self.assertLessEqual(len(unique_weights), number_of_clusters)


if __name__ == '__main__':
  test.main()
