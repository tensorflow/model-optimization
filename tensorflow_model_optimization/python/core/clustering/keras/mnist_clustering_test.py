# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for a simple convnet with clusterable layer on the MNIST dataset."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config

tf.random.set_seed(42)

keras = tf.keras

EPOCHS = 7
EPOCHS_FINE_TUNING = 4
NUMBER_OF_CLUSTERS = 8


def _build_model():
  """Builds a simple CNN model."""
  i = tf.keras.layers.Input(shape=(28, 28), name='input')
  x = tf.keras.layers.Reshape((28, 28, 1))(i)
  x = tf.keras.layers.Conv2D(
      filters=12, kernel_size=(3, 3), activation='relu', name='conv1')(
          x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Flatten()(x)
  output = tf.keras.layers.Dense(units=10)(x)

  model = tf.keras.Model(inputs=[i], outputs=[output])
  return model


def _get_dataset():
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  # Use subset of 60000 examples to keep unit test speed fast.
  x_train = x_train[0:1000]
  y_train = y_train[0:1000]
  return (x_train, y_train), (x_test, y_test)


def _train_model(model):
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

  (x_train, y_train), _ = _get_dataset()

  model.fit(x_train, y_train, epochs=EPOCHS)


def _cluster_model(model, number_of_clusters):

  (x_train, y_train), _ = _get_dataset()

  clustering_params = {
      'number_of_clusters':
          number_of_clusters,
      'cluster_centroids_init':
          cluster_config.CentroidInitialization.KMEANS_PLUS_PLUS
  }

  # Cluster model
  clustered_model = cluster.cluster_weights(model, **clustering_params)

  # Use smaller learning rate for fine-tuning
  # clustered model
  opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

  clustered_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=opt,
      metrics=['accuracy'])

  # Fine-tune clustered model
  clustered_model.fit(x_train, y_train, epochs=EPOCHS_FINE_TUNING)

  stripped_model = cluster.strip_clustering(clustered_model)
  stripped_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=opt,
      metrics=['accuracy'])

  return stripped_model


def _get_number_of_unique_weights(stripped_model, layer_nr, weights_nr):
  weights_as_list = stripped_model.layers[layer_nr].get_weights(
  )[weights_nr].reshape(-1,).tolist()
  nr_of_unique_weights = len(set(weights_as_list))

  return nr_of_unique_weights


class FunctionalTest(tf.test.TestCase):

  def testMnist(self):
    """In this test we test that 'kernel' weights are clustered."""
    model = _build_model()
    _train_model(model)

    # Checks that number of original weights('kernel') is greater than the
    # number of clusters
    nr_of_unique_weights = _get_number_of_unique_weights(model, -1, 0)
    self.assertGreater(nr_of_unique_weights, NUMBER_OF_CLUSTERS)

    # Record the number of unique values of 'bias'
    nr_of_bias_weights = _get_number_of_unique_weights(model, -1, 1)
    self.assertGreater(nr_of_bias_weights, NUMBER_OF_CLUSTERS)

    _, (x_test, y_test) = _get_dataset()

    results_original = model.evaluate(x_test, y_test)
    self.assertGreater(results_original[1], 0.8)

    clustered_model = _cluster_model(model, NUMBER_OF_CLUSTERS)

    results = clustered_model.evaluate(x_test, y_test)

    self.assertGreater(results[1], 0.8)

    nr_of_unique_weights = _get_number_of_unique_weights(clustered_model, -1, 0)
    self.assertLessEqual(nr_of_unique_weights, NUMBER_OF_CLUSTERS)

    # checks that we don't cluster 'bias' weights
    clustered_nr_of_bias_weights = _get_number_of_unique_weights(
        clustered_model, -1, 1)
    self.assertEqual(nr_of_bias_weights, clustered_nr_of_bias_weights)


if __name__ == '__main__':
  tf.test.main()
