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
"""Tests for weight clustering algorithm."""

import os
import tempfile
import zipfile

import tensorflow as tf

from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.common.keras.compression.algorithms import weight_clustering
from tensorflow_model_optimization.python.core.keras.compat import keras


def _build_model():
  i = keras.layers.Input(shape=(28, 28), name='input')
  x = keras.layers.Reshape((28, 28, 1))(i)
  x = keras.layers.Conv2D(
      20, 5, activation='relu', padding='valid', name='conv1'
  )(x)
  x = keras.layers.MaxPool2D(2, 2)(x)
  x = keras.layers.Conv2D(
      50, 5, activation='relu', padding='valid', name='conv2'
  )(x)
  x = keras.layers.MaxPool2D(2, 2)(x)
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dense(500, activation='relu', name='fc1')(x)
  output = keras.layers.Dense(10, name='fc2')(x)

  model = keras.Model(inputs=[i], outputs=[output])
  return model


def _get_dataset():
  mnist = keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  # Use subset of 60000 examples to keep unit test speed fast.
  x_train = x_train[:1000]
  y_train = y_train[:1000]

  return (x_train, y_train), (x_test, y_test)


def _train_model(model):
  loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
  (x_train, y_train), _ = _get_dataset()
  model.fit(x_train, y_train, epochs=1)


def _save_as_saved_model(model):
  saved_model_dir = tempfile.mkdtemp()
  model.save(saved_model_dir, include_optimizer=False)
  return saved_model_dir


def _get_zipped_directory_size(directory):
  """Measures the compressed size of a directory."""
  with tempfile.TemporaryFile(suffix='.zip') as zipped_file:
    for root, _, files in os.walk(directory):
      for file in files:
        with zipfile.ZipFile(
            zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
          f.write(os.path.join(root, file),
                  os.path.relpath(os.path.join(root, file),
                                  os.path.join(directory, '..')))

    zipped_file.seek(0, 2)
    return os.fstat(zipped_file.fileno()).st_size


class FunctionalTest(tf.test.TestCase):

  def testWeightClustering_TrainingE2E(self):
    number_of_clusters = 8
    model = _build_model()
    _train_model(model)
    original_saved_model_dir = _save_as_saved_model(model)

    algorithm = weight_clustering.WeightClustering(
        number_of_clusters=number_of_clusters,
        cluster_centroids_init=\
        cluster_config.CentroidInitialization.DENSITY_BASED)
    compressed_model = algorithm.compress_model(model)

    _train_model(compressed_model)

    saved_model_dir = _save_as_saved_model(compressed_model)

    _, (x_test, y_test) = _get_dataset()

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    compressed_model.compile(
        optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    results = compressed_model.evaluate(x_test, y_test)

    # Accuracy test.
    self.assertGreater(results[1], 0.85)  # 0.8708

    original_size = _get_zipped_directory_size(original_saved_model_dir)
    compressed_size = _get_zipped_directory_size(saved_model_dir)

    # Compressed model size test.
    self.assertLess(compressed_size, original_size / 4.0)

  def testWeightClustering_SingleLayer(self):
    number_of_clusters = 8
    i = keras.layers.Input(shape=(2), name='input')
    output = keras.layers.Dense(3, name='fc1')(i)
    model = keras.Model(inputs=[i], outputs=[output])

    dense_layer_weights = model.layers[1].get_weights()

    algorithm = weight_clustering.WeightClustering(
        number_of_clusters=number_of_clusters,
        cluster_centroids_init=\
        cluster_config.CentroidInitialization.DENSITY_BASED)
    compressed_model = algorithm.compress_model(model)

    dense_layer_compressed_weights = compressed_model.layers[1].get_weights()

    # clustering_centroids.
    self.assertEqual(
        dense_layer_compressed_weights[0].shape, (number_of_clusters,))

    # pulling_indices.
    self.assertEqual(
        dense_layer_compressed_weights[1].shape,
        dense_layer_weights[0].shape)
    self.assertEqual(str(dense_layer_compressed_weights[1].dtype), 'int64')
    self.assertAllInRange(
        dense_layer_compressed_weights[1], 0, number_of_clusters - 1)

    # bias
    assert (dense_layer_weights[1] == dense_layer_compressed_weights[2]).all()


if __name__ == '__main__':
  tf.test.main()
