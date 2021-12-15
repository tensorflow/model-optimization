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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config

tf.random.set_seed(42)

keras = tf.keras

EPOCHS = 7
EPOCHS_FINE_TUNING = 4
NUMBER_OF_CLUSTERS = 8
NUMBER_OF_CHANNELS = 12


def _build_model():
  """Builds a simple CNN model."""
  i = tf.keras.layers.Input(shape=(28, 28), name='input')
  x = tf.keras.layers.Reshape((28, 28, 1))(i)
  x = tf.keras.layers.Conv2D(
      filters=NUMBER_OF_CHANNELS, kernel_size=(3, 3),
      activation='relu', name='conv1')(x)
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


def _cluster_model(model,
                   number_of_clusters,
                   preserve_sparsity=False,
                   cluster_per_channel=False):

  (x_train, y_train), _ = _get_dataset()

  clustering_params = {
      'number_of_clusters':
          number_of_clusters,
      'cluster_centroids_init':
          cluster_config.CentroidInitialization.KMEANS_PLUS_PLUS,
      'cluster_per_channel':
          cluster_per_channel,
      'preserve_sparsity':
          preserve_sparsity
  }

  # Cluster model
  clustered_model = cluster.cluster_weights(model,
                                            **clustering_params)

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


def _get_number_of_unique_weights(stripped_model, layer_nr, weight_name):
  layer = stripped_model.layers[layer_nr]
  weight = getattr(layer, weight_name)
  weights_as_list = weight.numpy().reshape(-1,).tolist()
  nr_of_unique_weights = len(set(weights_as_list))

  return nr_of_unique_weights


def _deepcopy_model(model):
  model_copy = keras.models.clone_model(model)
  model_copy.set_weights(model.get_weights())
  return model_copy


class FunctionalTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(FunctionalTest, self).setUp()
    model = _build_model()
    _train_model(model)
    self.model = model
    self.dataset = _get_dataset()

  @parameterized.parameters(
      (False, False),
      (True, False),
      (True, True),
      (False, True)
  )
  def testMnist(self, preserve_sparsity, cluster_per_channel):
    """In this test we test that 'kernel' weights are clustered."""
    model = self.model
    _, (x_test, y_test) = self.dataset

    # Indices of Conv2D and Dense layers, respectively.
    layer_indices = [2, 5]

    # Dict to store the layer bias weight counts to
    # ensure they aren't clustered
    nr_of_bias_weights = {}

    # Checks that number of original weights ('kernel') and biases
    # are greater than the number of clusters for all clusterable layers
    for i in layer_indices:
      nr_of_unique_weights = _get_number_of_unique_weights(model, i, 'kernel')
      self.assertGreater(nr_of_unique_weights, NUMBER_OF_CLUSTERS)

      nr_of_bias_weights[i] = _get_number_of_unique_weights(model, i, 'bias')
      self.assertGreater(nr_of_bias_weights[i], NUMBER_OF_CLUSTERS)

    results_original = model.evaluate(x_test, y_test)
    self.assertGreater(results_original[1], 0.8)

    model_copy = _deepcopy_model(model)
    clustered_model = _cluster_model(model_copy, NUMBER_OF_CLUSTERS,
                                     preserve_sparsity=preserve_sparsity,
                                     cluster_per_channel=cluster_per_channel)

    results = clustered_model.evaluate(x_test, y_test)

    self.assertGreater(results[1], 0.8)

    for i in layer_indices:
      nr_of_unique_weights = _get_number_of_unique_weights(
          clustered_model, i, 'kernel')
      if (cluster_per_channel
          and isinstance(clustered_model.layers[i], tf.keras.layers.Conv2D)):
        self.assertLessEqual(nr_of_unique_weights,
                             NUMBER_OF_CLUSTERS * NUMBER_OF_CHANNELS)
      else:
        self.assertLessEqual(nr_of_unique_weights, NUMBER_OF_CLUSTERS)

      # checks that we don't cluster 'bias' weights
      clustered_nr_of_bias_weights = _get_number_of_unique_weights(
          clustered_model, i, 'bias')
      self.assertEqual(nr_of_bias_weights[i], clustered_nr_of_bias_weights)


if __name__ == '__main__':
  tf.test.main()
