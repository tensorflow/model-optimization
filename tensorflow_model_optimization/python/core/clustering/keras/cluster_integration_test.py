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

import os
import tempfile

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras.experimental import cluster as experimental_cluster

keras = tf.keras
layers = keras.layers
test = tf.test

CentroidInitialization = cluster_config.CentroidInitialization


class ClusterIntegrationTest(test.TestCase, parameterized.TestCase):
  """Integration tests for clustering."""

  def setUp(self):
    super(ClusterIntegrationTest, self).setUp()
    self.params = {
        "number_of_clusters": 8,
        "cluster_centroids_init": CentroidInitialization.LINEAR,
    }

    self.x_train = np.array(
        [[0.0, 1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 2.0, 3.0, 4.0],
         [0.0, 3.0, 2.0, 3.0, 4.0], [4.0, 1.0, 2.0, 3.0, 4.0],
         [5.0, 1.0, 2.0, 3.0, 4.0]],
        dtype="float32",
    )

    self.y_train = np.array(
        [[0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 2.0, 3.0, 4.0],
         [1.0, 0.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0],
         [0.0, 1.0, 2.0, 3.0, 4.0]],
        dtype="float32",
    )

    self.x_test = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0],
         [1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 1.0, 2.0, 3.0, 4.0],
         [9.0, 1.0, 0.0, 3.0, 0.0]],
        dtype="float32",
    )

    self.x_train2 = np.array(
        [[0.0, 1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 2.0, 3.0, 4.0],
         [0.0, 3.0, 2.0, 3.0, 4.0], [4.0, 1.0, 2.0, 3.0, 4.0],
         [5.0, 1.0, 2.0, 3.0, 4.0]],
        dtype="float32",
    )

    self.y_train2 = np.array(
        [[0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 2.0, 3.0, 4.0],
         [1.0, 0.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0],
         [0.0, 1.0, 2.0, 3.0, 4.0]],
        dtype="float32",
    )

  def dataset_generator(self):
    for x, y in zip(self.x_train, self.y_train):
      yield np.array([x]), np.array([y])

  def dataset_generator2(self):
    for x, y in zip(self.x_train2, self.y_train2):
      yield np.array([x]), np.array([y])

  def end_to_end_testing(self, original_model, clusters_check=None):
    """Test End to End clustering."""

    clustered_model = cluster.cluster_weights(original_model, **self.params)

    clustered_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"],
    )

    clustered_model.fit(x=self.dataset_generator(), steps_per_epoch=1)
    stripped_model = cluster.strip_clustering(clustered_model)
    if clusters_check is not None:
      clusters_check(stripped_model)

    _, tflite_file = tempfile.mkstemp(".tflite")
    _, keras_file = tempfile.mkstemp(".h5")

    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
    tflite_model = converter.convert()

    with open(tflite_file, "wb") as f:
      f.write(tflite_model)

    self._verify_tflite(tflite_file, self.x_test)

    os.remove(keras_file)
    os.remove(tflite_file)

  @staticmethod
  def _verify_tflite(tflite_file, x_test):
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    x = x_test[0]
    x = x.reshape((1,) + x.shape)
    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    interpreter.get_tensor(output_index)

  @keras_parameterized.run_all_keras_modes
  def testValuesRemainClusteredAfterTraining(self):
    """Verifies that training a clustered model does not destroy the clusters."""
    original_model = keras.Sequential([
        layers.Dense(5, input_shape=(5,)),
        layers.Dense(5),
    ])

    clustered_model = cluster.cluster_weights(original_model, **self.params)

    clustered_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"],
    )

    clustered_model.fit(x=self.dataset_generator(), steps_per_epoch=1)
    stripped_model = cluster.strip_clustering(clustered_model)
    weights_as_list = stripped_model.get_weights()[0].reshape(-1,).tolist()
    unique_weights = set(weights_as_list)
    self.assertLessEqual(len(unique_weights), self.params["number_of_clusters"])

  @keras_parameterized.run_all_keras_modes
  def testSparsityIsPreservedDuringTraining(self):
    # Set a specific random seed to ensure that we get some null weights to
    # test sparsity preservation with.
    tf.random.set_seed(1)

    # Verifies that training a clustered model does not destroy the sparsity of
    # the weights.
    original_model = keras.Sequential([
        layers.Dense(5, input_shape=(5,)),
        layers.Dense(5),
    ])

    # Using a mininum number of centroids to make it more likely that some
    # weights will be zero.
    clustering_params = {
        "number_of_clusters": 3,
        "cluster_centroids_init": CentroidInitialization.LINEAR,
        "preserve_sparsity": True
    }

    clustered_model = experimental_cluster.cluster_weights(
        original_model, **clustering_params)

    stripped_model_before_tuning = cluster.strip_clustering(clustered_model)
    weights_before_tuning = stripped_model_before_tuning.layers[0].kernel
    non_zero_weight_indices_before_tuning = np.nonzero(weights_before_tuning)

    clustered_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"],
    )
    clustered_model.fit(x=self.dataset_generator2(), steps_per_epoch=1)

    stripped_model_after_tuning = cluster.strip_clustering(clustered_model)
    weights_after_tuning = stripped_model_after_tuning.layers[0].kernel
    non_zero_weight_indices_after_tuning = np.nonzero(weights_after_tuning)
    weights_as_list_after_tuning = weights_after_tuning.numpy().reshape(
        -1,).tolist()
    unique_weights_after_tuning = set(weights_as_list_after_tuning)

    # Check that the null weights stayed the same before and after tuning.
    self.assertTrue(
        np.array_equal(non_zero_weight_indices_before_tuning,
                       non_zero_weight_indices_after_tuning))

    # Check that the number of unique weights matches the number of clusters.
    self.assertLessEqual(
        len(unique_weights_after_tuning), self.params["number_of_clusters"])

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def testEndToEndSequential(self):
    """Test End to End clustering - sequential model."""
    original_model = keras.Sequential([
        layers.Dense(5, input_shape=(5,)),
        layers.Dense(5),
    ])

    def clusters_check(stripped_model):
      # dense layer
      weights_as_list = stripped_model.get_weights()[0].reshape(-1,).tolist()
      unique_weights = set(weights_as_list)
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

    self.end_to_end_testing(original_model, clusters_check)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def testEndToEndFunctional(self):
    """Test End to End clustering - functional model."""
    inputs = keras.layers.Input(shape=(5,))
    layer1 = keras.layers.Dense(5)(inputs)
    layer2 = keras.layers.Dense(5)(layer1)
    original_model = keras.Model(inputs=inputs, outputs=layer2)

    def clusters_check(stripped_model):
      # First dense layer
      weights_as_list = stripped_model.get_weights()[0].reshape(-1,).tolist()
      unique_weights = set(weights_as_list)
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

    self.end_to_end_testing(original_model, clusters_check)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def testEndToEndDeepLayer(self):
    """Test End to End clustering for the model with deep layer."""
    internal_model = tf.keras.Sequential(
        [tf.keras.layers.Dense(5, input_shape=(5,))])
    original_model = keras.Sequential([
        internal_model,
        layers.Dense(5),
    ])

    def clusters_check(stripped_model):
      # inner dense layer
      weights_as_list = (
          stripped_model.submodules[1].trainable_weights[0].numpy().flatten())
      unique_weights = set(weights_as_list)
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

      # outer dense layer
      weights_as_list = (
          stripped_model.submodules[4].trainable_weights[0].numpy().flatten())
      unique_weights = set(weights_as_list)
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

    self.end_to_end_testing(original_model, clusters_check)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def testEndToEndDeepLayer2(self):
    """Test End to End clustering for the model with 2 deep layers."""
    internal_model = tf.keras.Sequential(
        [tf.keras.layers.Dense(5, input_shape=(5,))])
    intermediate_model = keras.Sequential([
        internal_model,
        layers.Dense(5),
    ])
    original_model = keras.Sequential([
        intermediate_model,
        layers.Dense(5),
    ])

    def clusters_check(stripped_model):
      # first inner dense layer
      weights_as_list = (
          stripped_model.submodules[1].trainable_weights[0].numpy().flatten())
      unique_weights = set(weights_as_list)
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

      # second inner dense layer
      weights_as_list = (
          stripped_model.submodules[4].trainable_weights[0].numpy().flatten())
      unique_weights = set(weights_as_list)
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

      # outer dense layer
      weights_as_list = (
          stripped_model.submodules[7].trainable_weights[0].numpy().flatten())
      unique_weights = set(weights_as_list)
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

    self.end_to_end_testing(original_model, clusters_check)

  @keras_parameterized.run_all_keras_modes
  def testWeightsAreLearningDuringClustering(self):
    """Verifies that weights are updated during training a clustered model.

    Training a clustered model should update original_weights,
    clustered_centroids and bias.
    """
    original_model = keras.Sequential([layers.Dense(5, input_shape=(5,))])

    clustered_model = cluster.cluster_weights(original_model, **self.params)

    clustered_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"],
    )

    class CheckWeightsCallback(keras.callbacks.Callback):

      def on_train_batch_begin(self, batch, logs=None):
        # Save weights before batch
        self.original_weight_kernel = (
            self.model.layers[0].original_clusterable_weights["kernel"].numpy())
        self.cluster_centroids_kernel = (
            self.model.layers[0].cluster_centroids["kernel"].numpy())
        self.bias = (self.model.layers[0].layer.bias.numpy())

      def on_train_batch_end(self, batch, logs=None):
        # Check weights are different after batch
        assert not np.array_equal(
            self.original_weight_kernel,
            self.model.layers[0].original_clusterable_weights["kernel"].numpy())
        assert not np.array_equal(
            self.cluster_centroids_kernel,
            self.model.layers[0].cluster_centroids["kernel"].numpy())
        assert not np.array_equal(self.bias,
                                  self.model.layers[0].layer.bias.numpy())

    clustered_model.fit(
        x=self.dataset_generator(),
        steps_per_epoch=5,
        callbacks=[CheckWeightsCallback()])


if __name__ == "__main__":
  test.main()
