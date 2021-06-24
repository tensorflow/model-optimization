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
"""Integration tests for CQAT, PCQAT cases."""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras.experimental import cluster as experimental_cluster
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras.collaborative_optimizations.cluster_preserve import (
    default_8bit_cluster_preserve_quantize_scheme,)
from tensorflow_model_optimization.python.core.quantization.keras.collaborative_optimizations.cluster_preserve.cluster_utils import (
    strip_clustering_cqat,)

layers = tf.keras.layers


@keras_parameterized.run_all_keras_modes
class ClusterPreserveIntegrationTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ClusterPreserveIntegrationTest, self).setUp()
    self.cluster_params = {
        'number_of_clusters': 4,
        'cluster_centroids_init': cluster_config.CentroidInitialization.LINEAR
    }

  def compile_and_fit(self, model):
    """Here we compile and fit the model."""
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy'],
    )
    model.fit(
        np.random.rand(20, 10),
        tf.keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        batch_size=20)

  def _get_number_of_unique_weights(self, stripped_model, layer_nr,
                                    weight_name):
    layer = stripped_model.layers[layer_nr]
    if isinstance(layer, quantize.quantize_wrapper.QuantizeWrapper):
      for weight_item in layer.trainable_weights:
        if weight_name in weight_item.name:
          weight = weight_item
    else:
      weight = getattr(layer, weight_name)
    weights_as_list = weight.numpy().flatten()
    nr_of_unique_weights = len(set(weights_as_list))
    return nr_of_unique_weights

  def _get_sparsity(self, model):
    sparsity_list = []
    for layer in model.layers:
      for weights in layer.trainable_weights:
        if 'kernel' in weights.name:
          np_weights = tf.keras.backend.get_value(weights)
          sparsity = 1.0 - np.count_nonzero(np_weights) / float(
              np_weights.size)
          sparsity_list.append(sparsity)

    return sparsity_list

  def _get_clustered_model(self, preserve_sparsity):
    """Cluster the (sparse) model and return clustered_model."""
    tf.random.set_seed(1)
    original_model = tf.keras.Sequential([
        layers.Dense(5, activation='softmax', input_shape=(10,)),
        layers.Flatten(),
    ])

    # Manually set sparsity in the Dense layer if preserve_sparsity is on
    if preserve_sparsity:
      first_layer_weights = original_model.layers[0].get_weights()
      first_layer_weights[0][:][0:2] = 0.0
      original_model.layers[0].set_weights(first_layer_weights)

    # Start the sparsity-aware clustering
    clustering_params = {
        'number_of_clusters': 4,
        'cluster_centroids_init': cluster_config.CentroidInitialization.LINEAR,
        'preserve_sparsity': True
    }

    clustered_model = experimental_cluster.cluster_weights(
        original_model, **clustering_params)

    return clustered_model

  def _pcqat_training(self, preserve_sparsity, quant_aware_annotate_model):
    """PCQAT training on the input model."""
    quant_aware_model = quantize.quantize_apply(
        quant_aware_annotate_model,
        scheme=default_8bit_cluster_preserve_quantize_scheme
        .Default8BitClusterPreserveQuantizeScheme(preserve_sparsity))

    self.compile_and_fit(quant_aware_model)

    stripped_pcqat_model = strip_clustering_cqat(quant_aware_model)

    # Check the unique weights of clustered_model and pcqat_model
    # layer 0 is the quantize_layer
    num_of_unique_weights_pcqat = self._get_number_of_unique_weights(
        stripped_pcqat_model, 1, 'kernel')

    sparsity_pcqat = self._get_sparsity(stripped_pcqat_model)

    return sparsity_pcqat, num_of_unique_weights_pcqat

  def testEndToEndClusterPreserve(self):
    """Runs CQAT end to end and whole model is quantized."""
    original_model = tf.keras.Sequential([
        layers.Dense(5, activation='softmax', input_shape=(10,))
    ])
    clustered_model = cluster.cluster_weights(
        original_model,
        **self.cluster_params)
    self.compile_and_fit(clustered_model)
    clustered_model = cluster.strip_clustering(clustered_model)
    num_of_unique_weights_clustering = self._get_number_of_unique_weights(
        clustered_model, 0, 'kernel')

    quant_aware_annotate_model = (
        quantize.quantize_annotate_model(clustered_model))

    quant_aware_model = quantize.quantize_apply(
        quant_aware_annotate_model,
        scheme=default_8bit_cluster_preserve_quantize_scheme
        .Default8BitClusterPreserveQuantizeScheme())

    self.compile_and_fit(quant_aware_model)
    stripped_cqat_model = strip_clustering_cqat(quant_aware_model)

    # Check the unique weights of a certain layer of
    # clustered_model and pcqat_model
    num_of_unique_weights_cqat = self._get_number_of_unique_weights(
        stripped_cqat_model, 1, 'kernel')
    self.assertAllEqual(num_of_unique_weights_clustering,
                        num_of_unique_weights_cqat)

  def testEndToEndClusterPreservePerLayer(self):
    """Runs CQAT end to end and model is quantized per layers."""
    original_model = tf.keras.Sequential([
        layers.Dense(5, activation='relu', input_shape=(10,)),
        layers.Dense(5, activation='softmax', input_shape=(10,))
    ])
    clustered_model = cluster.cluster_weights(
        original_model,
        **self.cluster_params)
    self.compile_and_fit(clustered_model)
    clustered_model = cluster.strip_clustering(clustered_model)
    num_of_unique_weights_clustering = self._get_number_of_unique_weights(
        clustered_model, 1, 'kernel')

    def apply_quantization_to_dense(layer):
      if isinstance(layer, tf.keras.layers.Dense):
        return quantize.quantize_annotate_layer(layer)
      return layer

    quant_aware_annotate_model = tf.keras.models.clone_model(
        clustered_model,
        clone_function=apply_quantization_to_dense,
    )

    quant_aware_model = quantize.quantize_apply(
        quant_aware_annotate_model,
        scheme=default_8bit_cluster_preserve_quantize_scheme
        .Default8BitClusterPreserveQuantizeScheme())

    self.compile_and_fit(quant_aware_model)
    stripped_cqat_model = strip_clustering_cqat(
        quant_aware_model)

    # Check the unique weights of a certain layer of
    # clustered_model and pcqat_model
    num_of_unique_weights_cqat = self._get_number_of_unique_weights(
        stripped_cqat_model, 2, 'kernel')
    self.assertAllEqual(num_of_unique_weights_clustering,
                        num_of_unique_weights_cqat)

  def testEndToEndClusterPreserveOneLayer(self):
    """Runs CQAT end to end and model is quantized only for a single layer."""
    original_model = tf.keras.Sequential([
        layers.Dense(5, activation='relu', input_shape=(10,)),
        layers.Dense(5, activation='softmax', input_shape=(10,), name='qat')
    ])
    clustered_model = cluster.cluster_weights(
        original_model,
        **self.cluster_params)
    self.compile_and_fit(clustered_model)
    clustered_model = cluster.strip_clustering(clustered_model)
    num_of_unique_weights_clustering = self._get_number_of_unique_weights(
        clustered_model, 1, 'kernel')

    def apply_quantization_to_dense(layer):
      if isinstance(layer, tf.keras.layers.Dense):
        if layer.name == 'qat':
          return quantize.quantize_annotate_layer(layer)
      return layer

    quant_aware_annotate_model = tf.keras.models.clone_model(
        clustered_model,
        clone_function=apply_quantization_to_dense,
    )

    quant_aware_model = quantize.quantize_apply(
        quant_aware_annotate_model,
        scheme=default_8bit_cluster_preserve_quantize_scheme
        .Default8BitClusterPreserveQuantizeScheme())

    self.compile_and_fit(quant_aware_model)

    stripped_cqat_model = strip_clustering_cqat(
        quant_aware_model)

    # Check the unique weights of a certain layer of
    # clustered_model and pcqat_model
    num_of_unique_weights_cqat = self._get_number_of_unique_weights(
        stripped_cqat_model, 1, 'kernel')
    self.assertAllEqual(num_of_unique_weights_clustering,
                        num_of_unique_weights_cqat)

  def testEndToEndPruneClusterPreserveQAT(self):
    """Runs PCQAT end to end when we quantize the whole model."""
    preserve_sparsity = True
    clustered_model = self._get_clustered_model(preserve_sparsity)
    # Save the kernel weights
    first_layer_weights = clustered_model.layers[0].weights[1]
    stripped_model_before_tuning = cluster.strip_clustering(
        clustered_model)
    nr_of_unique_weights_before = self._get_number_of_unique_weights(
        stripped_model_before_tuning, 0, 'kernel')

    self.compile_and_fit(clustered_model)

    stripped_model_clustered = cluster.strip_clustering(clustered_model)
    weights_after_tuning = stripped_model_clustered.layers[0].kernel
    nr_of_unique_weights_after = self._get_number_of_unique_weights(
        stripped_model_clustered, 0, 'kernel')

    # Check after sparsity-aware clustering, despite zero centroid can drift,
    # the final number of unique weights remains the same
    self.assertEqual(nr_of_unique_weights_before, nr_of_unique_weights_after)

    # Check that the zero weights stayed the same before and after tuning.
    # There might be new weights that become zeros but sparsity-aware
    # clustering preserves the original zero weights in the original positions
    # of the weight array
    self.assertTrue(
        np.array_equal(first_layer_weights[:][0:2],
                       weights_after_tuning[:][0:2]))

    # Check sparsity before the input of PCQAT
    sparsity_pruning = self._get_sparsity(stripped_model_clustered)

    # PCQAT: when the preserve_sparsity flag is True, the PCQAT should work
    quant_aware_annotate_model = (
        quantize.quantize_annotate_model(stripped_model_clustered)
    )

    # When preserve_sparsity is True in PCQAT, the final sparsity of
    # the layer stays the same or larger than that of the input layer
    preserve_sparsity = True
    sparsity_pcqat, unique_weights_pcqat = self._pcqat_training(
        preserve_sparsity, quant_aware_annotate_model)
    self.assertAllGreaterEqual(np.array(sparsity_pcqat),
                               sparsity_pruning[0])
    self.assertAllEqual(nr_of_unique_weights_after, unique_weights_pcqat)

  def testPassingNonPrunedModelToPCQAT(self):
    """Runs PCQAT as CQAT if the input model is not pruned."""
    preserve_sparsity = False
    clustered_model = self._get_clustered_model(preserve_sparsity)

    clustered_model = cluster.strip_clustering(clustered_model)
    nr_of_unique_weights_after = self._get_number_of_unique_weights(
        clustered_model, 0, 'kernel')

    # Check after plain clustering, if there are no zero weights,
    # PCQAT falls back to CQAT
    quant_aware_annotate_model = (
        quantize.quantize_annotate_model(clustered_model)
    )

    quant_aware_model = quantize.quantize_apply(
        quant_aware_annotate_model,
        scheme=default_8bit_cluster_preserve_quantize_scheme
        .Default8BitClusterPreserveQuantizeScheme(True))

    self.compile_and_fit(quant_aware_model)
    stripped_pcqat_model = strip_clustering_cqat(
        quant_aware_model)

    # Check the unique weights of clustered_model and pcqat_model
    num_of_unique_weights_pcqat = self._get_number_of_unique_weights(
        stripped_pcqat_model, 1, 'kernel')
    self.assertAllEqual(nr_of_unique_weights_after,
                        num_of_unique_weights_pcqat)

  @parameterized.parameters((0.), (2.))
  def testPassingModelWithUniformWeightsToPCQAT(self, uniform_weights):
    """If pruned_clustered_model has uniform weights, it won't break PCQAT."""
    preserve_sparsity = True
    original_model = tf.keras.Sequential([
        layers.Dense(5, activation='softmax', input_shape=(10,)),
        layers.Flatten(),
    ])

    # Manually set all weights to the same value in the Dense layer
    first_layer_weights = original_model.layers[0].get_weights()
    first_layer_weights[0][:] = uniform_weights
    original_model.layers[0].set_weights(first_layer_weights)

    # Start the sparsity-aware clustering
    clustering_params = {
        'number_of_clusters': 4,
        'cluster_centroids_init': cluster_config.CentroidInitialization.LINEAR,
        'preserve_sparsity': True
    }

    clustered_model = experimental_cluster.cluster_weights(
        original_model, **clustering_params)
    clustered_model = cluster.strip_clustering(clustered_model)

    nr_of_unique_weights_after = self._get_number_of_unique_weights(
        clustered_model, 0, 'kernel')
    sparsity_pruning = self._get_sparsity(clustered_model)

    quant_aware_annotate_model = (
        quantize.quantize_annotate_model(clustered_model)
    )

    sparsity_pcqat, unique_weights_pcqat = self._pcqat_training(
        preserve_sparsity, quant_aware_annotate_model)
    self.assertAllGreaterEqual(np.array(sparsity_pcqat),
                               sparsity_pruning[0])
    self.assertAllEqual(nr_of_unique_weights_after, unique_weights_pcqat)

  def testTrainableWeightsBehaveCorrectlyDuringPCQAT(self):
    """PCQAT zero centroid masks stay the same and trainable variables are updating between epochs."""
    preserve_sparsity = True
    clustered_model = self._get_clustered_model(preserve_sparsity)
    clustered_model = cluster.strip_clustering(clustered_model)

    # Apply PCQAT
    quant_aware_annotate_model = (
        quantize.quantize_annotate_model(clustered_model)
    )

    quant_aware_model = quantize.quantize_apply(
        quant_aware_annotate_model,
        scheme=default_8bit_cluster_preserve_quantize_scheme
        .Default8BitClusterPreserveQuantizeScheme(True))

    quant_aware_model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy'],
    )

    class CheckCentroidsAndTrainableVarsCallback(tf.keras.callbacks.Callback):
      """Check the updates of trainable variables and centroid masks."""
      def on_epoch_begin(self, batch, logs=None):
        # Check cluster centroids have the zero in the right position
        vars_dictionary = self.model.layers[1]._weight_vars[0][2]
        self.centroid_mask = vars_dictionary['centroids_mask']
        self.zero_centroid_index_begin = np.where(
            self.centroid_mask == 0)[0]

        # Check trainable weights before training
        self.layer_kernel = (
            self.model.layers[1].weights[3].numpy()
        )
        self.original_weight = vars_dictionary['ori_weights_vars_tf'].numpy()
        self.centroids = vars_dictionary['cluster_centroids_tf'].numpy()

      def on_epoch_end(self, batch, logs=None):
        # Check the index of the zero centroids are not changed after training
        vars_dictionary = self.model.layers[1]._weight_vars[0][2]
        self.zero_centroid_index_end = np.where(
            vars_dictionary['centroids_mask'] == 0)[0]
        assert np.array_equal(
            self.zero_centroid_index_begin,
            self.zero_centroid_index_end
        )

        # Check trainable variables after training are updated
        assert not np.array_equal(
            self.layer_kernel,
            self.model.layers[1].weights[3].numpy()
        )
        assert not np.array_equal(
            self.original_weight,
            vars_dictionary['ori_weights_vars_tf'].numpy()
        )
        assert not np.array_equal(
            self.centroids,
            vars_dictionary['cluster_centroids_tf'].numpy()
        )

    # Use many epochs to verify layer's kernel weights are updating because
    # they can stay the same after being trained using only the first batch
    # of data for instance
    quant_aware_model.fit(np.random.rand(20, 10),
                          tf.keras.utils.to_categorical(
                              np.random.randint(5, size=(20, 1)), 5),
                          steps_per_epoch=5,
                          epochs=3,
                          callbacks=[CheckCentroidsAndTrainableVarsCallback()])


if __name__ == '__main__':
  tf.test.main()
