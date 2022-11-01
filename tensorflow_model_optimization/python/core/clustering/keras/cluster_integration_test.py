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

  def _batch(self, dims, batch_size):
    if dims[0] is None:
      dims[0] = batch_size
    return dims

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

  def _verify_tflite(self, tflite_file, x_test):
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    x = x_test[0]
    x = x.reshape((1,) + x.shape)
    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    interpreter.get_tensor(output_index)

  def _get_number_of_unique_weights(self, stripped_model, layer_nr,
                                    weight_name):
    layer = stripped_model.layers[layer_nr]
    weight = getattr(layer, weight_name)
    weights_as_list = weight.numpy().flatten()
    nr_of_unique_weights = len(set(weights_as_list))
    return nr_of_unique_weights

  def testDefaultClusteringInit(self):
    """Verifies that default initialization method is KMEANS_PLUS_PLUS."""
    original_model = keras.Sequential([
        layers.Dense(5, input_shape=(5,)),
        layers.Dense(5),
    ])
    clustered_model = cluster.cluster_weights(original_model, **self.params)

    for i in range(len(clustered_model.layers)):
      if hasattr(clustered_model.layers[i], "layer"):
        init_method = clustered_model.layers[i].get_config(
        )["cluster_centroids_init"]
        self.assertEqual(init_method, CentroidInitialization.KMEANS_PLUS_PLUS)

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

  def testSparsityIsPreservedDuringTraining(self):
    """Set a specific random seed.

       Ensures that we get some null weights to test sparsity preservation with.
    """
    tf.random.set_seed(1)
    # Verifies that training a clustered model with null weights in it
    # does not destroy the sparsity of the weights.
    original_model = keras.Sequential([
        layers.Dense(5, input_shape=(5,)),
        layers.Flatten(),
    ])
    # Reset the kernel weights to reflect potential zero drifting of
    # the cluster centroids
    first_layer_weights = original_model.layers[0].get_weights()
    first_layer_weights[0][:][0:2] = 0.0
    first_layer_weights[0][:][3] = [-0.13, -0.08, -0.05, 0.005, 0.13]
    first_layer_weights[0][:][4] = [-0.13, -0.08, -0.05, 0.005, 0.13]
    original_model.layers[0].set_weights(first_layer_weights)
    clustering_params = {
        "number_of_clusters": 6,
        "preserve_sparsity": True
    }
    clustered_model = experimental_cluster.cluster_weights(
        original_model, **clustering_params)
    stripped_model_before_tuning = cluster.strip_clustering(clustered_model)
    nr_of_unique_weights_before = self._get_number_of_unique_weights(
        stripped_model_before_tuning, 0, "kernel")
    clustered_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"],
    )
    clustered_model.fit(x=self.dataset_generator(), steps_per_epoch=100)
    stripped_model_after_tuning = cluster.strip_clustering(clustered_model)
    weights_after_tuning = stripped_model_after_tuning.layers[0].kernel
    nr_of_unique_weights_after = self._get_number_of_unique_weights(
        stripped_model_after_tuning, 0, "kernel")
    # Check after sparsity-aware clustering, despite zero centroid can drift,
    # the final number of unique weights remains the same
    self.assertLessEqual(nr_of_unique_weights_after,
                         nr_of_unique_weights_before)
    # Check that the null weights stayed the same before and after tuning.
    # There might be new weights that become zeros but sparsity-aware
    # clustering preserves the original null weights in the original positions
    # of the weight array
    self.assertTrue(
        np.array_equal(first_layer_weights[0][:][0:2],
                       weights_after_tuning[:][0:2]))
    # Check that the number of unique weights matches the number of clusters.
    self.assertLessEqual(
        nr_of_unique_weights_after,
        clustering_params["number_of_clusters"])

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

  def testEndToEndConv1DAndConv1DTranspose(self):
    """Test End to End clustering - model with Conv1D and Conv1DTranspose."""
    inp = layers.Input(batch_shape=(1, 16))
    x = layers.Conv1D(
        10, 16, 4, padding="valid", use_bias=False)(
            tf.expand_dims(inp, axis=-1))
    y = layers.Conv1DTranspose(1, 16, 4, padding="valid", use_bias=False)(x)
    model = keras.models.Model(inputs=inp, outputs=[y])

    def apply_clustering(layer):
      if isinstance(layer, keras.layers.Conv1D) or isinstance(
          layer, keras.layers.Conv1DTranspose):
        return cluster.cluster_weights(layer, **self.params)
      return layer

    model_to_cluster = keras.models.clone_model(
        model,
        clone_function=apply_clustering,
    )

    model_to_cluster.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"]
    )
    model_to_cluster.fit(
        np.random.randn(*self._batch(model.input.get_shape().as_list(), 16)),
        np.random.randn(*self._batch(model.output.get_shape().as_list(), 16)),
        steps_per_epoch=1)
    clustered_model = cluster.strip_clustering(model_to_cluster)

    def do_checks(layer, layer_name):
      self.assertEqual(layer.name, layer_name)
      unique_weights = np.unique(layer.weights[0].numpy().flatten())
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

    do_checks(clustered_model.layers[2], "conv1d")
    do_checks(clustered_model.layers[3], "conv1d_transpose")

  @parameterized.parameters(
      (False, 16),  # number_of_clusters > Conv2D filters
      (True,
       8),  # number_of_clusters < Conv2D filters (but clustering by channel)
      (True, 12),  # number_of_clusters = Conv2D filters
      (False, 12),  # number_of_clusters = Conv2D filters
  )
  def testEndToEnd1x1Conv2d(self, cluster_per_channel, number_of_clusters):
    """Test End to End clustering - model with 1x1 Conv2D.

    Clustering should not be performed at all, since number of
    weights in the layer is too low in all of these cases.

    Args:
      cluster_per_channel: An optional boolean value.
      number_of_clusters: A number of cluster centroids to form clusters.
    """
    kernel_size = (1, 1)

    inp = keras.layers.Input(shape=(28, 28), batch_size=16)
    x = keras.layers.Reshape(target_shape=(28, 28, 1))(inp)
    x = keras.layers.Conv2D(
        filters=12, kernel_size=kernel_size, activation=tf.nn.relu)(
            x)
    model = keras.models.Model(inputs=inp, outputs=[x])

    cluster_params = {
        "number_of_clusters": number_of_clusters,
        "cluster_per_channel": cluster_per_channel
    }

    # Get unique kernel weights on original model for comparison
    original_unique_weights = model.layers[2].weights[0]

    def apply_clustering(layer):
      if isinstance(layer, keras.layers.Conv2D):
        return cluster.cluster_weights(layer, **cluster_params)
      return layer

    # Ensure a warning is given to the user that clustering is not
    # implemented for this layer
    with self.assertWarnsRegex(Warning,
                               r"Layer conv2d does not have enough weights"):
      model_to_cluster = keras.models.clone_model(
          model,
          clone_function=apply_clustering,
      )

    model_to_cluster.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"])
    model_to_cluster.fit(
        np.random.randn(*self._batch(model.input.get_shape().as_list(), 16)),
        np.random.randn(*self._batch(model.output.get_shape().as_list(), 16)),
        steps_per_epoch=1)
    clustered_model = cluster.strip_clustering(model_to_cluster)

    def do_checks(layer, layer_name, original_unique_weights):
      self.assertEqual(layer.name, layer_name)
      unique_weights = layer.weights[0]

      # Ensure clustering was not performed on the 1x1 Conv
      # (weights are identical to original unclustered layer)
      self.assertAllEqual(unique_weights, original_unique_weights)

    do_checks(clustered_model.layers[2], "conv2d", original_unique_weights)

  def testStripClusteringSequentialModelWithRegulariser(self):
    """Verifies that stripping the clustering wrappers from a sequential model produces the expected config."""
    original_model = keras.Sequential([
        layers.Dense(5, input_shape=(5,)),
        layers.Dense(5, kernel_regularizer=tf.keras.regularizers.L1(0.01)),
    ])

    def clusters_check(stripped_model):
      # dense layer
      weights_as_list = stripped_model.get_weights()[0].reshape(-1,).tolist()
      unique_weights = set(weights_as_list)
      self.assertLessEqual(
          len(unique_weights), self.params["number_of_clusters"])

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
        # Skip the first batch since the update from the start of training is
        # too small.
        if not batch:
          return
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


class ClusterRNNIntegrationTest(tf.test.TestCase, parameterized.TestCase):
  """Integration tests for clustering RNN layers."""

  def setUp(self):
    super(ClusterRNNIntegrationTest, self).setUp()
    self.max_features = 10
    self.maxlen = 2
    self.batch_size = 32
    self.x_train = np.random.random((64, self.maxlen))
    self.y_train = np.random.randint(0, 2, (64,))

    self.params_clustering = {
        "number_of_clusters": 16,
    }

  def _train(self, model):
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.fit(
        self.x_train,
        self.y_train,
        batch_size=self.batch_size,
        epochs=1,
    )

  def _clusterTrainStrip(self, model):
    clustered_model = cluster.cluster_weights(
        model,
        **self.params_clustering,
    )
    self._train(clustered_model)
    stripped_model = cluster.strip_clustering(clustered_model)

    return stripped_model

  def _assertNbUniqueWeights(self, weight, expected_unique_weights):
    nr_unique_weights = len(np.unique(weight.numpy().flatten()))
    assert nr_unique_weights == expected_unique_weights

  def testClusterSimpleRNN(self):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(self.max_features, 16,
                                     input_length=self.maxlen))
    model.add(keras.layers.SimpleRNN(16, return_sequences=True))
    model.add(keras.layers.SimpleRNN(16))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("sigmoid"))

    stripped_model = self._clusterTrainStrip(model)

    self._assertNbUniqueWeights(
        weight=stripped_model.layers[1].cell.kernel,
        expected_unique_weights=self.params_clustering["number_of_clusters"],
    )
    self._assertNbUniqueWeights(
        weight=stripped_model.layers[1].cell.recurrent_kernel,
        expected_unique_weights=self.params_clustering["number_of_clusters"],
    )

    self._train(stripped_model)

  def testClusterLSTM(self):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(self.max_features, 16,
                                     input_length=self.maxlen))
    model.add(keras.layers.LSTM(16, return_sequences=True))
    model.add(keras.layers.LSTM(16))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("sigmoid"))

    stripped_model = self._clusterTrainStrip(model)

    self._assertNbUniqueWeights(
        weight=stripped_model.layers[1].cell.kernel,
        expected_unique_weights=self.params_clustering["number_of_clusters"],
    )
    self._assertNbUniqueWeights(
        weight=stripped_model.layers[1].cell.recurrent_kernel,
        expected_unique_weights=self.params_clustering["number_of_clusters"],
    )

    self._train(stripped_model)

  def testClusterGRU(self):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(self.max_features, 16,
                                     input_length=self.maxlen))
    model.add(keras.layers.GRU(16, return_sequences=True))
    model.add(keras.layers.GRU(16))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("sigmoid"))

    stripped_model = self._clusterTrainStrip(model)

    self._assertNbUniqueWeights(
        weight=stripped_model.layers[1].cell.kernel,
        expected_unique_weights=self.params_clustering["number_of_clusters"],
    )
    self._assertNbUniqueWeights(
        weight=stripped_model.layers[1].cell.recurrent_kernel,
        expected_unique_weights=self.params_clustering["number_of_clusters"],
    )

    self._train(stripped_model)

  def testClusterBidirectional(self):
    model = keras.models.Sequential()
    model.add(
        keras.layers.Embedding(self.max_features, 16, input_length=self.maxlen))
    model.add(keras.layers.Bidirectional(keras.layers.SimpleRNN(16)))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("sigmoid"))

    self._clusterTrainStrip(model)

    self._assertNbUniqueWeights(
        weight=model.layers[1].forward_layer.cell.kernel,
        expected_unique_weights=self.params_clustering["number_of_clusters"],
    )
    self._assertNbUniqueWeights(
        weight=model.layers[1].forward_layer.cell.recurrent_kernel,
        expected_unique_weights=self.params_clustering["number_of_clusters"],
    )
    self._assertNbUniqueWeights(
        weight=model.layers[0].embeddings,
        expected_unique_weights=self.params_clustering["number_of_clusters"],
    )

  def testClusterStackedRNNCells(self):
    model = keras.models.Sequential()
    model.add(
        keras.layers.Embedding(self.max_features, 16, input_length=self.maxlen))
    model.add(
        tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells(
                [keras.layers.SimpleRNNCell(16) for _ in range(2)])))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("sigmoid"))

    self._clusterTrainStrip(model)

    self._assertNbUniqueWeights(
        weight=model.layers[1].cell.cells[0].kernel,
        expected_unique_weights=self.params_clustering["number_of_clusters"],
    )
    self._assertNbUniqueWeights(
        weight=model.layers[1].cell.cells[0].recurrent_kernel,
        expected_unique_weights=self.params_clustering["number_of_clusters"],
    )
    self._assertNbUniqueWeights(
        weight=model.layers[0].embeddings,
        expected_unique_weights=self.params_clustering["number_of_clusters"],
    )


class ClusterMHAIntegrationTest(tf.test.TestCase, parameterized.TestCase):
  """Integration tests for clustering MHA layer."""

  def setUp(self):
    super(ClusterMHAIntegrationTest, self).setUp()
    self.x_train = np.random.uniform(size=(500, 32, 32))
    self.y_train = np.random.randint(low=0, high=1024, size=(500,))

    self.nr_of_clusters = 16
    self.params_clustering = {
        "number_of_clusters": self.nr_of_clusters,
        "cluster_centroids_init": CentroidInitialization.KMEANS_PLUS_PLUS,
    }

  def _get_model(self):
    """Returns functional model with MHA layer."""
    inp = tf.keras.layers.Input(shape=(32, 32), batch_size=100)
    x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)(
        query=inp, value=inp)
    out = tf.keras.layers.Flatten()(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

  def testMHA(self):
    model = self._get_model()

    clustered_model = cluster.cluster_weights(model, **self.params_clustering)

    clustered_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    clustered_model.fit(
        self.x_train, self.y_train, epochs=1, batch_size=100, verbose=1)

    stripped_model = cluster.strip_clustering(clustered_model)

    layer_mha = stripped_model.layers[1]
    for weight in layer_mha.weights:
      if "kernel" in weight.name:
        nr_unique_weights = len(np.unique(weight.numpy()))
        assert nr_unique_weights == self.nr_of_clusters


class ClusterPerChannelIntegrationTest(tf.test.TestCase,
                                       parameterized.TestCase):
  """Integration test for cluster_per_channel for Conv2D layer."""

  def setUp(self):
    super(ClusterPerChannelIntegrationTest, self).setUp()
    self.x_train = np.random.uniform(size=(500, 32, 32))
    self.y_train = np.random.randint(low=0, high=1024, size=(500,))

    self.nr_of_clusters = 4
    self.num_channels = 12
    self.params_clustering = {
        "number_of_clusters": self.nr_of_clusters,
        "cluster_centroids_init": CentroidInitialization.KMEANS_PLUS_PLUS,
        "cluster_per_channel": True
    }

  def _get_model(self):
    """Returns functional model with Conv2D layer."""
    inp = tf.keras.layers.Input(shape=(32, 32), batch_size=100)
    x = tf.keras.layers.Reshape((32, 32, 1))(inp)
    x = tf.keras.layers.Conv2D(
        filters=self.num_channels, kernel_size=(3, 3),
        activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2, 2)(x)
    out = tf.keras.layers.Flatten()(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

  def testPerChannel(self):
    model = self._get_model()

    clustered_model = cluster.cluster_weights(model, **self.params_clustering)

    clustered_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    clustered_model.fit(
        self.x_train, self.y_train, epochs=2, batch_size=100, verbose=1)

    stripped_model = cluster.strip_clustering(clustered_model)

    layer_conv2d = stripped_model.layers[2]
    layer_conv2d_kernel_weight = None
    for weight in layer_conv2d.weights:
      if "kernel" in weight.name:
        layer_conv2d_kernel_weight = weight
    self.assertIsNotNone(layer_conv2d_kernel_weight)
    nr_unique_weights = len(np.unique(layer_conv2d_kernel_weight.numpy()))
    self.assertEqual(nr_unique_weights, self.nr_of_clusters*self.num_channels)

    # The above check is too general.
    # We need to check that we actually have nr_of_clusters per channel.
    # If more general case passed, then we do tighter check.
    # Note that we assume that data_format is NHWC.
    for i in range(self.num_channels):
      nr_unique_weights_per_channel = len(
          np.unique(layer_conv2d_kernel_weight[:, :, :, i]))
      self.assertEqual(nr_unique_weights_per_channel, self.nr_of_clusters)


if __name__ == "__main__":
  test.main()
