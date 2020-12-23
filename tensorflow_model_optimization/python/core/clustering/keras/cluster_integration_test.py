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

class SubclassedModel(keras.Model):
  """Subclassed model with one layer."""

  def __init__(self):
    """Subclassed model with one dense layer."""
    super(SubclassedModel, self).__init__(name='subclass_model')
    self.dense_layer = keras.layers.Dense(5, activation='relu')

  def call(self, inputs):
    return self.dense_layer(inputs)

class SubclassedModelTwoLayers(keras.Model):
  """Subclassed model with two layers."""

  def __init__(self):
    """Subclassed model with two layers."""
    super(SubclassedModelTwoLayers, self).__init__(name='subclass_model')
    self.dense_layer1 = keras.layers.Dense(5, activation='relu')
    self.dense_layer2 = keras.layers.Dense(5, activation='softmax')

  def call(self, inputs):
    x = self.dense_layer1(inputs)
    return self.dense_layer2(x)

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
    """Set a specific random seed to ensure that we get some null weights to test sparsity preservation with."""
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
    weights_before_tuning = stripped_model_before_tuning.get_weights()[0]
    non_zero_weight_indices_before_tuning = np.nonzero(weights_before_tuning)

    clustered_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"],
    )
    clustered_model.fit(x=self.dataset_generator2(), steps_per_epoch=1)

    stripped_model_after_tuning = cluster.strip_clustering(clustered_model)
    weights_after_tuning = stripped_model_after_tuning.get_weights()[0]
    non_zero_weight_indices_after_tuning = np.nonzero(weights_after_tuning)
    weights_as_list_after_tuning = weights_after_tuning.reshape(-1,).tolist()
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

  def testStripClusteringSequentialModelWithRegulariser(self):
    """
    Verifies that stripping the clustering wrappers from a sequential model
    produces the expected config.
    """
    original_model = keras.Sequential([
        layers.Dense(5, input_shape=(5,)),
        layers.Dense(5, kernel_regularizer=tf.keras.regularizers.L1(0.01)),
    ])

    def clusters_check(stripped_model):
      # dense layer
      weights_as_list = stripped_model.get_weights()[0].reshape(-1,).tolist()
      unique_weights = set(weights_as_list)
      self.assertLessEqual(len(unique_weights), self.params["number_of_clusters"])

    self.end_to_end_testing(original_model, clusters_check)

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
  def testEndToEndSubclassedModel(self):
    """Test End to End clustering for the subclassed model.
    In this test we pass the whole subclassed model for clustering.
    We check that the number of weights is less the requested
    number of clusters after stripping clustering wrapper.

    """
    subclassed_model = SubclassedModel()

    clustered_model = cluster.cluster_weights(subclassed_model, **self.params)

    clustered_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"]
    )

    # The model should be trained a little bit.
    clustered_model.fit(x=self.dataset_generator(), steps_per_epoch=1)
    stripped_model = cluster.strip_clustering(clustered_model)

    nr_unique_weights = len(np.unique(stripped_model.layers[0].\
        trainable_weights[0].numpy().flatten()))
    self.assertLessEqual(nr_unique_weights, self.params["number_of_clusters"])

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def testEndToEndSubclassedModelTwoLayers(self):
    """Test End to End clustering for the subclass model.

    This test demonstrates another approach.
    All layers that are present in the subclassed model
    (see SubclassedModelTwoLayers definition above) are wrapped
    manually. The model should be re-build in this case.

    We need to strip clustering away manually as well (see how it is
    done inside the test).

    Clustering is working well and clusters are updated during
    training."""
    subclassed_model = SubclassedModelTwoLayers()
    input_shape = (1, 5)

    # We need to build the model
    subclassed_model.build(input_shape=input_shape)

    # Check that the number of weights is bigger than the number of clusters.
    nr_unique_weights = len(np.unique(subclassed_model.layers[0].\
        trainable_weights[0].numpy().flatten()))
    self.assertGreater(nr_unique_weights, self.params["number_of_clusters"])
    nr_unique_weights = len(np.unique(subclassed_model.layers[1].\
        trainable_weights[0].numpy().flatten()))
    self.assertGreater(nr_unique_weights, self.params["number_of_clusters"])

    # Now we apply cluster_weights for each layer.
    subclassed_model.dense_layer1 = cluster.cluster_weights(
      subclassed_model.dense_layer1, **self.params)
    subclassed_model.dense_layer2 = cluster.cluster_weights(
      subclassed_model.dense_layer2, **self.params)

    # We need to re-build the model again.
    subclassed_model.build(input_shape=input_shape)

    subclassed_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"]
      )

    subclassed_model.fit(x=self.dataset_generator(), steps_per_epoch=1)

    # We strip from layers that were wrapped.
    subclassed_model.dense_layer1 = cluster.strip_clustering(subclassed_model.dense_layer1)
    subclassed_model.dense_layer2 = cluster.strip_clustering(subclassed_model.dense_layer2)

    # Checks that the number of unique values is less than the requested
    # number of clusters.
    nr_unique_weights = len(np.unique(subclassed_model.layers[0].\
        trainable_weights[0].numpy().flatten()))
    self.assertLessEqual(nr_unique_weights, self.params["number_of_clusters"])
    nr_unique_weights = len(np.unique(subclassed_model.layers[1].\
        trainable_weights[0].numpy().flatten()))
    self.assertLessEqual(nr_unique_weights, self.params["number_of_clusters"])

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def testEndToEndSubclassedModelAsDeepLayer(self):
    """Test End to End clustering for the model with the layer as a subclass model."""
    # This case is not supported currently.

    internal_model = tf.keras.Sequential([tf.keras.layers.Dense(5, input_shape=(5,))])
    subclassed_model = SubclassedModel()
    original_model = keras.Sequential([
        internal_model,
        subclassed_model,
    ])

    with self.assertRaisesRegexp(ValueError, "Subclassed models.*"):
      self.end_to_end_testing(original_model)

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
      weights_as_list = stripped_model.submodules[1].trainable_weights[0].\
        numpy().flatten()
      unique_weights = set(weights_as_list)
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

      # outer dense layer
      weights_as_list = stripped_model.submodules[4].trainable_weights[0].\
        numpy().flatten()
      unique_weights = set(weights_as_list)
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

    self.end_to_end_testing(original_model, clusters_check)

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
      weights_as_list = stripped_model.submodules[1].trainable_weights[0].\
        numpy().flatten()
      unique_weights = set(weights_as_list)
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

      # second inner dense layer
      weights_as_list = stripped_model.submodules[4].\
        trainable_weights[0].\
        numpy().flatten()
      unique_weights = set(weights_as_list)
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

      # outer dense layer
      weights_as_list = stripped_model.submodules[7].trainable_weights[0].\
        numpy().flatten()
      unique_weights = set(weights_as_list)
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

    self.end_to_end_testing(original_model, clusters_check)


if __name__ == "__main__":
  test.main()
