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

  def _get_conv_model(self,
                      nr_of_channels,
                      data_format=None,
                      kernel_size=(3, 3)):
    """Returns functional model with Conv2D layer."""
    inp = tf.keras.layers.Input(shape=(32, 32), batch_size=100)
    shape = (1, 32, 32) if data_format == 'channels_first' else (32, 32, 1)
    x = tf.keras.layers.Reshape(shape)(inp)
    x = tf.keras.layers.Conv2D(
        filters=nr_of_channels,
        kernel_size=kernel_size,
        data_format=data_format,
        activation='relu')(
            x)
    x = tf.keras.layers.MaxPool2D(2, 2)(x)
    out = tf.keras.layers.Flatten()(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

  def _compile_and_fit_conv_model(self, model, nr_epochs=1):
    """Compile and fit conv model from _get_conv_model."""
    x_train = np.random.uniform(size=(500, 32, 32))
    y_train = np.random.randint(low=0, high=1024, size=(500,))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

    model.fit(x_train, y_train, epochs=nr_epochs, batch_size=100, verbose=1)

    return model

  def _get_conv_clustered_model(self,
                                nr_of_channels,
                                nr_of_clusters,
                                data_format,
                                preserve_sparsity,
                                kernel_size=(3, 3)):
    """Returns clustered per channel model with Conv2D layer."""
    tf.random.set_seed(42)
    model = self._get_conv_model(nr_of_channels, data_format, kernel_size)

    if preserve_sparsity:
      # Make the convolutional layer sparse by nullifying half of weights
      assert model.layers[2].name == 'conv2d'

      conv_layer_weights = model.layers[2].get_weights()
      shape = conv_layer_weights[0].shape
      conv_layer_weights_flatten = conv_layer_weights[0].flatten()

      nr_elems = len(conv_layer_weights_flatten)
      conv_layer_weights_flatten[0:1 + nr_elems // 2] = 0.0
      pruned_conv_layer_weights = tf.reshape(conv_layer_weights_flatten, shape)
      conv_layer_weights[0] = pruned_conv_layer_weights
      model.layers[2].set_weights(conv_layer_weights)

    clustering_params = {
        'number_of_clusters':
            nr_of_clusters,
        'cluster_centroids_init':
            cluster_config.CentroidInitialization.KMEANS_PLUS_PLUS,
        'cluster_per_channel':
            True,
        'preserve_sparsity':
            preserve_sparsity
    }

    clustered_model = experimental_cluster.cluster_weights(model,
                                                           **clustering_params)
    clustered_model = self._compile_and_fit_conv_model(clustered_model)

    # Returns un-stripped model
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

  def testEndToEndClusterPreserveQATClusteredPerChannel(
      self, data_format='channels_last'):
    """Runs CQAT end to end for the model that is clustered per channel."""

    nr_of_channels = 12
    nr_of_clusters = 4

    clustered_model = self._get_conv_clustered_model(
        nr_of_channels, nr_of_clusters, data_format, preserve_sparsity=False)
    stripped_model = cluster.strip_clustering(clustered_model)

    # Save the kernel weights
    conv2d_layer = stripped_model.layers[2]
    self.assertEqual(conv2d_layer.name, 'conv2d')

    # should be nr_of_channels * nr_of_clusters
    nr_unique_weights = -1

    for weight in conv2d_layer.weights:
      if 'kernel' in weight.name:
        nr_unique_weights = len(np.unique(weight.numpy()))
        self.assertLessEqual(nr_unique_weights, nr_of_clusters*nr_of_channels)

    quant_aware_annotate_model = (
        quantize.quantize_annotate_model(stripped_model)
    )

    quant_aware_model = quantize.quantize_apply(
        quant_aware_annotate_model,
        scheme=default_8bit_cluster_preserve_quantize_scheme
        .Default8BitClusterPreserveQuantizeScheme())

    # Lets train for more epochs to have a chance to scatter clusters
    model = self._compile_and_fit_conv_model(quant_aware_model, 3)

    stripped_cqat_model = strip_clustering_cqat(model)

    # Check the unique weights of a certain layer of
    # clustered_model and pcqat_model
    layer_nr = 3
    num_of_unique_weights_cqat = self._get_number_of_unique_weights(
        stripped_cqat_model, layer_nr, 'kernel')
    self.assertLessEqual(num_of_unique_weights_cqat, nr_unique_weights)

    # We need to do tighter check: we check that the number of unique
    # weights per channel is less than the given nr_of_channels
    layer = stripped_cqat_model.layers[layer_nr]
    weight_to_check = None
    if isinstance(layer, quantize.quantize_wrapper.QuantizeWrapper):
      for weight_item in layer.trainable_weights:
        if 'kernel' in weight_item.name:
          weight_to_check = weight_item

    assert weight_to_check is not None

    for i in range(nr_of_channels):
      nr_unique_weights_per_channel = len(
          np.unique(weight_to_check[:, :, :, i]))
      assert nr_unique_weights_per_channel == nr_of_clusters

  def testEndToEndPCQATClusteredPerChannel(self, data_format='channels_last'):
    """Runs PCQAT end to end for the model that is clustered per channel."""

    nr_of_channels = 12
    nr_of_clusters = 4

    clustered_model = self._get_conv_clustered_model(
        nr_of_channels, nr_of_clusters, data_format, preserve_sparsity=True)
    stripped_model = cluster.strip_clustering(clustered_model)

    # Save the kernel weights
    conv2d_layer = stripped_model.layers[2]
    self.assertEqual(conv2d_layer.name, 'conv2d')

    # should be nr_of_channels * nr_of_clusters
    nr_unique_weights = -1

    for weight in conv2d_layer.weights:
      if 'kernel' in weight.name:
        nr_unique_weights = len(np.unique(weight.numpy()))
        self.assertLessEqual(nr_unique_weights, nr_of_clusters*nr_of_channels)

    # get sparsity before PCQAT training
    # we expect that only one value will be returned
    control_sparsity = self._get_sparsity(stripped_model)
    self.assertGreater(control_sparsity[0], 0.5)

    quant_aware_annotate_model = (
        quantize.quantize_annotate_model(stripped_model)
    )

    quant_aware_model = quantize.quantize_apply(
        quant_aware_annotate_model,
        scheme=default_8bit_cluster_preserve_quantize_scheme
        .Default8BitClusterPreserveQuantizeScheme())

    # Lets train for more epochs to have a chance to scatter clusters
    model = self._compile_and_fit_conv_model(quant_aware_model, 3)

    stripped_cqat_model = strip_clustering_cqat(model)

    # Check the unique weights of a certain layer of
    # clustered_model and cqat_model
    layer_nr = 3
    num_of_unique_weights_cqat = self._get_number_of_unique_weights(
        stripped_cqat_model, layer_nr, 'kernel')
    self.assertLessEqual(num_of_unique_weights_cqat, nr_unique_weights)

    # We need to do tighter check: we check that the number of unique
    # weights per channel is less than the given nr_of_channels
    layer = stripped_cqat_model.layers[layer_nr]
    weight_to_check = None
    if isinstance(layer, quantize.quantize_wrapper.QuantizeWrapper):
      for weight_item in layer.trainable_weights:
        if 'kernel' in weight_item.name:
          weight_to_check = weight_item

    assert weight_to_check is not None

    for i in range(nr_of_channels):
      nr_unique_weights_per_channel = len(
          np.unique(weight_to_check[:, :, :, i]))
      assert nr_unique_weights_per_channel == nr_of_clusters

    cqat_sparsity = self._get_sparsity(stripped_cqat_model)
    self.assertLessEqual(cqat_sparsity[0], control_sparsity[0])

  def testEndToEndPCQATClusteredPerChannelConv2d1x1(self,
                                                    data_format='channels_last'
                                                    ):
    """Runs PCQAT for model containing a 1x1 Conv2D.

    (with insufficient number of weights per channel).

    Args:
      data_format: Format of input data.
    """
    nr_of_channels = 12
    nr_of_clusters = 4

    # Ensure a warning is given to the user that
    # clustering is not implemented for this layer
    with self.assertWarnsRegex(Warning,
                               r'Layer conv2d does not have enough weights'):
      clustered_model = self._get_conv_clustered_model(
          nr_of_channels,
          nr_of_clusters,
          data_format,
          preserve_sparsity=True,
          kernel_size=(1, 1))
      stripped_model = cluster.strip_clustering(clustered_model)

    # Save the kernel weights
    conv2d_layer = stripped_model.layers[2]
    self.assertEqual(conv2d_layer.name, 'conv2d')

    for weight in conv2d_layer.weights:
      if 'kernel' in weight.name:
        # Original number of unique weights
        nr_original_weights = len(np.unique(weight.numpy()))
        self.assertLess(nr_original_weights, nr_of_channels * nr_of_clusters)

        # Demonstrate unmodified test layer has less weights
        # than requested clusters
        for channel in range(nr_of_channels):
          channel_weights = (
              weight[:, channel, :, :]
              if data_format == 'channels_first' else weight[:, :, :, channel])
          nr_channel_weights = len(channel_weights)
          self.assertGreater(nr_channel_weights, 0)
          self.assertLessEqual(nr_channel_weights, nr_of_clusters)

    # get sparsity before PCQAT training
    # we expect that only one value will be returned
    control_sparsity = self._get_sparsity(stripped_model)
    self.assertGreater(control_sparsity[0], 0.5)

    quant_aware_annotate_model = (
        quantize.quantize_annotate_model(stripped_model))

    with self.assertWarnsRegex(
        Warning, r'No clustering performed on layer quant_conv2d'):
      quant_aware_model = quantize.quantize_apply(
          quant_aware_annotate_model,
          scheme=default_8bit_cluster_preserve_quantize_scheme
          .Default8BitClusterPreserveQuantizeScheme(preserve_sparsity=True))

    # Lets train for more epochs to have a chance to scatter clusters
    model = self._compile_and_fit_conv_model(quant_aware_model, 3)

    stripped_cqat_model = strip_clustering_cqat(model)

    # Check the unique weights of a certain layer of
    # clustered_model and cqat_model, ensuring unchanged
    layer_nr = 3
    num_of_unique_weights_cqat = self._get_number_of_unique_weights(
        stripped_cqat_model, layer_nr, 'kernel')
    self.assertEqual(num_of_unique_weights_cqat, nr_original_weights)

    cqat_sparsity = self._get_sparsity(stripped_cqat_model)
    self.assertLessEqual(cqat_sparsity[0], control_sparsity[0])

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
