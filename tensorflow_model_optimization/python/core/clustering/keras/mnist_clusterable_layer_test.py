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
from tensorflow_model_optimization.python.core.clustering.keras import clusterable_layer
from tensorflow_model_optimization.python.core.clustering.keras import clustering_algorithm

tf.random.set_seed(42)

keras = tf.keras

EPOCHS = 7
EPOCHS_FINE_TUNING = 4
NUMBER_OF_CLUSTERS = 8


class MyDenseLayer(keras.layers.Dense, clusterable_layer.ClusterableLayer):

  def get_clusterable_weights(self):
    # Cluster kernel and bias.
    return [('kernel', self.kernel), ('bias', self.bias)]


class ClusterableWeightsCA(clustering_algorithm.AbstractClusteringAlgorithm):
  """This class provides a special lookup function for the the weights 'w'.

  It reshapes and tile centroids the same way as the weights. This allows us
  to find pulling indices efficiently.
  """

  def get_pulling_indices(self, weight):
    clst_num = self.cluster_centroids.shape[0]
    tiled_weights = tf.tile(tf.expand_dims(weight, axis=2), [1, 1, clst_num])
    tiled_cluster_centroids = tf.tile(
        tf.reshape(self.cluster_centroids, [1, 1, clst_num]),
        [weight.shape[0], weight.shape[1], 1])

    # We find the nearest cluster centroids and store them so that ops can build
    # their kernels upon it
    pulling_indices = tf.argmin(
        tf.abs(tiled_weights - tiled_cluster_centroids), axis=2)

    return pulling_indices


class MyClusterableLayer(keras.layers.Layer,
                         clusterable_layer.ClusterableLayer):

  def __init__(self, units=32):
    super(MyClusterableLayer, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(
        shape=(input_shape[-1], self.units),
        initializer='random_normal',
        trainable=True,
    )
    self.b = self.add_weight(
        shape=(self.units,), initializer='random_normal', trainable=False)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

  def get_clusterable_weights(self):
    # Cluster only weights 'w'
    return [('w', self.w)]

  def get_clusterable_algorithm(self, weight_name):
    """Returns clustering algorithm for the custom weights 'w'."""
    if weight_name == 'w':
      return ClusterableWeightsCA
    else:
      # We don't cluster other weights.
      return None


def _build_model():
  """Builds model with MyDenseLayer."""
  i = tf.keras.layers.Input(shape=(28, 28), name='input')
  x = tf.keras.layers.Reshape((28, 28, 1))(i)
  x = tf.keras.layers.Conv2D(
      filters=12, kernel_size=(3, 3), activation='relu', name='conv1')(
          x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Flatten()(x)
  output = MyDenseLayer(units=10)(x)

  model = tf.keras.Model(inputs=[i], outputs=[output])
  return model


def _build_model_2():
  """Builds model with MyClusterableLayer layer."""
  i = tf.keras.layers.Input(shape=(28, 28), name='input')
  x = tf.keras.layers.Reshape((28, 28, 1))(i)
  x = tf.keras.layers.Conv2D(
      filters=12, kernel_size=(3, 3), activation='relu', name='conv1')(
          x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Flatten()(x)
  output = MyClusterableLayer(units=10)(x)

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
          cluster_config.CentroidInitialization.DENSITY_BASED
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

  def testMnistMyDenseLayer(self):
    """Test model with a custom clusterable layer derived from Dense.

    This customerable layer (see MyDenseLayer definition above) provides the
    function get_clusterable_weights() so that both 'kernel' weights as well
    as 'bias' weights are clustered.
    """
    model = _build_model()
    _train_model(model)

    # Checks that number of original weights('kernel') is greater than the
    # number of clusters.
    nr_of_unique_weights = _get_number_of_unique_weights(model, -1, 0)
    self.assertGreater(nr_of_unique_weights, NUMBER_OF_CLUSTERS)

    # Checks that number of original weights('bias') is greater than the number
    # of clusters
    nr_of_unique_weights = _get_number_of_unique_weights(model, -1, 1)
    self.assertGreater(nr_of_unique_weights, NUMBER_OF_CLUSTERS)

    _, (x_test, y_test) = _get_dataset()

    results_original = model.evaluate(x_test, y_test)
    self.assertGreater(results_original[1], 0.8)

    clustered_model = _cluster_model(model, NUMBER_OF_CLUSTERS)

    results = clustered_model.evaluate(x_test, y_test)

    self.assertGreater(results[1], 0.8)

    # checks 'kernel' weights of the last layer: MyDenseLayer
    nr_of_unique_weights = _get_number_of_unique_weights(clustered_model, -1, 0)
    self.assertLessEqual(nr_of_unique_weights, NUMBER_OF_CLUSTERS)

    # checks 'bias' weights of the last layer: MyDenseLayer
    nr_of_unique_weights = _get_number_of_unique_weights(clustered_model, -1, 1)
    self.assertLessEqual(nr_of_unique_weights, NUMBER_OF_CLUSTERS)

  def testMnistClusterableLayer(self):
    """Test keras custom layer.

    We test the keras custom layer with the provided clustering algorithm
    (see MyClusterableLayer above). We cluster only 'w' weights and the class
    ClusterableWeightsCA provides the function get_pulling_indices for the
    layer-out of 'w' weights.

    We skip evaluation in this test as it takes some time.
    """
    model = _build_model_2()
    _train_model(model)

    # Checks that number of original weights 'w' is greater than the number
    # of clusters.
    nr_of_unique_weights = _get_number_of_unique_weights(model, -1, 0)
    self.assertGreater(nr_of_unique_weights, NUMBER_OF_CLUSTERS)

    clustered_model = _cluster_model(model, NUMBER_OF_CLUSTERS)

    # Checks clustered weights 'w'.
    nr_of_unique_weights = _get_number_of_unique_weights(clustered_model, -1, 0)
    self.assertLessEqual(nr_of_unique_weights, NUMBER_OF_CLUSTERS)


if __name__ == '__main__':
  tf.test.main()
