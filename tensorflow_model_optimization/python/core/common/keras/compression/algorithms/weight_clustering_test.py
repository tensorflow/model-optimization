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
import unittest

import tensorflow as tf

from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.common.keras.compression.algorithms import weight_clustering


def _build_model():
  i = tf.keras.layers.Input(shape=(28, 28), name='input')
  x = tf.keras.layers.Reshape((28, 28, 1))(i)
  x = tf.keras.layers.Conv2D(
      20, 5, activation='relu', padding='valid', name='conv1')(
          x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Conv2D(
      50, 5, activation='relu', padding='valid', name='conv2')(
          x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(500, activation='relu', name='fc1')(x)
  output = tf.keras.layers.Dense(10, name='fc2')(x)

  model = tf.keras.Model(inputs=[i], outputs=[output])
  return model


def _get_dataset():
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  # Use subset of 60000 examples to keep unit test speed fast.
  x_train = x_train[:1000]
  y_train = y_train[:1000]

  return (x_train, y_train), (x_test, y_test)


def _train_model(model):
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
  (x_train, y_train), _ = _get_dataset()
  model.fit(x_train, y_train, epochs=1)


def _save_as_saved_model(model):
  saved_model_dir = tempfile.mkdtemp()
  model.save(saved_model_dir)
  return saved_model_dir


def _get_directory_size_in_bytes(directory):
  total = 0
  try:
    for entry in os.scandir(directory):
      if entry.is_file():
        # if it's a file, use stat() function
        total += entry.stat().st_size
      elif entry.is_dir():
        # if it's a directory, recursively call this function
        total += _get_directory_size_in_bytes(entry.path)
  except NotADirectoryError:
    # if `directory` isn't a directory, get the file size then
    return os.path.getsize(directory)
  except PermissionError:
    # if for whatever reason we can't open the folder, return 0
    return 0
  return total


class FunctionalTest(tf.test.TestCase):

  # TODO(b/246652360): Re-Enable the test once it is fixed.
  @unittest.skip('Test needs to be fixed')
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

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    compressed_model.compile(
        optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    results = compressed_model.evaluate(x_test, y_test)

    # Accuracy test.
    self.assertGreater(results[1], 0.85)  # 0.8708

    original_size = _get_directory_size_in_bytes(original_saved_model_dir)
    compressed_size = _get_directory_size_in_bytes(saved_model_dir)

    # Compressed model size test.
    # TODO(tfmot): gzip compression can reduce file size much better.
    self.assertLess(compressed_size, original_size / 1.3)

  def testWeightClustering_SingleLayer(self):
    number_of_clusters = 8
    i = tf.keras.layers.Input(shape=(2), name='input')
    output = tf.keras.layers.Dense(3, name='fc1')(i)
    model = tf.keras.Model(inputs=[i], outputs=[output])

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
