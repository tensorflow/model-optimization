# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Clustering API functions for Keras models."""

import warnings

import tensorflow as tf

from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper
from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids

k = tf.keras.backend
CustomObjectScope = tf.keras.utils.CustomObjectScope
CentroidInitialization = cluster_config.CentroidInitialization
Layer = tf.keras.layers.Layer
InputLayer = tf.keras.layers.InputLayer


def cluster_scope():
  """Provides a scope in which Clustered layers and models can be deserialized.

  If a keras model or layer has been clustered, it needs to be within this scope
  to be successfully deserialized.

  Returns:
      Object of type `CustomObjectScope` with clustering objects included.

  Example:

  ```python
  clustered_model = cluster_weights(model, **self.params)
  tf.keras.models.save_model(clustered_model, keras_file)

  with cluster_scope():
    loaded_model = tf.keras.models.load_model(keras_file)
  ```
  """
  return CustomObjectScope({'ClusterWeights': cluster_wrapper.ClusterWeights})


def cluster_weights(
    to_cluster,
    number_of_clusters,
    cluster_centroids_init=CentroidInitialization.KMEANS_PLUS_PLUS,
    **kwargs):
  """Modifies a keras layer or model to be clustered during training.

  This function wraps a keras model or layer with clustering functionality
  which clusters the layer's weights during training. For examples, using
  this with number_of_clusters equals 8 will ensure that each weight tensor has
  no more than 8 unique values.

  Before passing to the clustering API, a model should already be trained and
  show some acceptable performance on the testing/validation sets.

  The function accepts either a single keras layer
  (subclass of `keras.layers.Layer`), list of keras layers or a keras model
  (instance of `keras.models.Model`) and handles them appropriately.

  If it encounters a layer it does not know how to handle, it will throw an
  error. While clustering an entire model, even a single unknown layer would
  lead to an error.

  Cluster a model:

  ```python
  clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
  }

  clustered_model = cluster_weights(original_model, **clustering_params)
  ```

  Cluster a layer:

  ```python
  clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
  }

  model = tf.keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      cluster_weights(layers.Dense(2, activation='tanh'), **clustering_params)
  ])
  ```

  Arguments:
      to_cluster: A single keras layer, list of keras layers, or a
        `tf.keras.Model` instance.
      number_of_clusters: the number of cluster centroids to form when
        clustering a layer/model. For example, if number_of_clusters=8 then only
        8 unique values will be used in each weight array.
      cluster_centroids_init: enum value that determines how the cluster
        centroids will be initialized.
        Can have following values:
          1. RANDOM : centroids are sampled using the uniform distribution
            between the minimum and maximum weight values in a given layer
          2. DENSITY_BASED : density-based sampling. First, cumulative
            distribution function is built for weights, then y-axis is evenly
            spaced into number_of_clusters regions. After this the corresponding
            x values are obtained and used to initialize clusters centroids.
          3. LINEAR : cluster centroids are evenly spaced between the minimum
            and maximum values of a given weight
      **kwargs: Additional keyword arguments to be passed to the keras layer.
        Ignored when to_cluster is not a keras layer.

  Returns:
    Layer or model modified to include clustering related metadata.

  Raises:
    ValueError: if the keras layer is unsupported, or the keras model contains
    an unsupported layer.
  """
  return _cluster_weights(to_cluster, number_of_clusters,
                          cluster_centroids_init, **kwargs)


def _cluster_weights(to_cluster,
                     number_of_clusters,
                     cluster_centroids_init,
                     preserve_sparsity=False,
                     cluster_per_channel=False,
                     **kwargs):
  """Modifies a keras layer or model to be clustered during training.

  This function wraps a keras model or layer with clustering functionality
  which clusters the layer's weights during training. For examples, using
  this with number_of_clusters equals 8 will ensure that each weight tensor has
  no more than 8 unique values.

  Before passing to the clustering API, a model should already be trained and
  show some acceptable performance on the testing/validation sets.

  The function accepts either a single keras layer
  (subclass of `keras.layers.Layer`), list of keras layers or a keras model
  (instance of `keras.models.Model`) and handles them appropriately.

  If it encounters a layer it does not know how to handle, it will throw an
  error. While clustering an entire model, even a single unknown layer would
  lead to an error.

  Cluster a model:

  ```python
  clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': CentroidInitialization.DENSITY_BASED,
    'cluster_per_channel': False,
    'preserve_sparsity': False
  }

  clustered_model = cluster_weights(original_model, **clustering_params)
  ```

  Cluster a layer:

  ```python
  clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': CentroidInitialization.DENSITY_BASED,
    'cluster_per_channel': False,
    'preserve_sparsity': False
  }

  model = tf.keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      cluster_weights(layers.Dense(2, activation='tanh'), **clustering_params)
  ])
  ```

  Cluster a layer with sparsity preservation (experimental):

  ```python
  clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': CentroidInitialization.DENSITY_BASED,
    'preserve_sparsity': True
  }

  model = tf.keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      cluster_weights(layers.Dense(2, activation='tanh'), **clustering_params)
  ])
  ```

  Arguments:
      to_cluster: A single keras layer, list of keras layers, or a
        `tf.keras.Model` instance.
      number_of_clusters: the number of cluster centroids to form when
        clustering a layer/model. For example, if number_of_clusters=8 then only
        8 unique values will be used in each weight array.
      cluster_centroids_init: `tfmot.clustering.keras.CentroidInitialization`
        instance that determines how the cluster centroids will be initialized.
      preserve_sparsity (experimental): optional boolean value that determines
        whether or not sparsity preservation will be enforced during training.
        When used along with cluster_per_channel flag below, the zero centroid
        is treated separately and maintained individually for each channel.
      cluster_per_channel: optional boolean value that determines whether the
        clustering should be applied separately on the individual channels, as
        opposed to the whole kernel. Only applicable to Conv2D layers and is
        ignored otherwise. The number of clusters in this case would be
        num_clusters*num_channels. This is useful for the collaborative
        optimization pipeline where clustering is followed by quantization,
        since Conv2D is quantized per-channel, so we end up with
        num_clusters*num_channels total clusters at the end. Clustering
        per-channel from the beginning leads to better accuracy.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
        Ignored when to_cluster is not a keras layer.

  Returns:
    Layer or model modified to include clustering related metadata.

  Raises:
    ValueError: if the keras layer is unsupported, or the keras model contains
    an unsupported layer.
  """
  if not clustering_centroids.CentroidsInitializerFactory.init_is_supported(
      cluster_centroids_init):
    raise ValueError('Cluster centroid initialization {} not supported'.format(
        cluster_centroids_init))

  def _add_clustering_wrapper(layer):
    if isinstance(layer, tf.keras.Model):
      # Check whether the model is a subclass.
      # NB: This check is copied from keras.py file in tensorflow.
      # There is no available public API to do this check.
      # pylint: disable=protected-access
      if (not layer._is_graph_network and
          not isinstance(layer, tf.keras.models.Sequential)):
        raise ValueError('Subclassed models are not supported currently.')

      return tf.keras.models.clone_model(
          layer, input_tensors=None, clone_function=_add_clustering_wrapper)
    if isinstance(layer, cluster_wrapper.ClusterWeights):
      return layer
    if isinstance(layer, InputLayer):
      return layer.__class__.from_config(layer.get_config())
    if isinstance(layer, tf.keras.layers.RNN) or isinstance(
        layer, tf.keras.layers.Bidirectional):
      return cluster_wrapper.ClusterWeightsRNN(
          layer,
          number_of_clusters,
          cluster_centroids_init,
          preserve_sparsity,
          **kwargs,
      )
    if isinstance(layer, tf.keras.layers.MultiHeadAttention):
      return cluster_wrapper.ClusterWeightsMHA(
          layer,
          number_of_clusters,
          cluster_centroids_init,
          preserve_sparsity,
          **kwargs,
      )

    # Skip clustering if Conv2D layer has insufficient number of weights
    # for type of clustering
    if isinstance(
        layer,
        tf.keras.layers.Conv2D) and not layer_has_enough_weights_to_cluster(
            layer, number_of_clusters, cluster_per_channel):
      return layer

    return cluster_wrapper.ClusterWeights(layer, number_of_clusters,
                                          cluster_centroids_init,
                                          preserve_sparsity,
                                          cluster_per_channel, **kwargs)

  def _wrap_list(layers):
    output = []
    for layer in layers:
      output.append(_add_clustering_wrapper(layer))

    return output

  if isinstance(to_cluster, tf.keras.Model):
    return tf.keras.models.clone_model(
        to_cluster, input_tensors=None, clone_function=_add_clustering_wrapper)
  if isinstance(to_cluster, Layer):
    return _add_clustering_wrapper(layer=to_cluster)
  if isinstance(to_cluster, list):
    return _wrap_list(to_cluster)


def strip_clustering(model):
  """Strips clustering wrappers from the model.

  Once a model has been clustered, this method can be used
  to restore the original model with the clustered weights.

  Only sequential and functional models are supported for now.

  Arguments:
      model: A `tf.keras.Model` instance with clustered layers.

  Returns:
    A keras model with clustering wrappers removed.

  Raises:
    ValueError: if the model is not a `tf.keras.Model` instance.
    NotImplementedError: if the model is a subclass model.

  Usage:

  ```python
  orig_model = tf.keras.Model(inputs, outputs)
  clustered_model = cluster_weights(orig_model)
  exported_model = strip_clustering(clustered_model)
  ```
  The exported_model and the orig_model have the same structure.
  """
  if not isinstance(model, tf.keras.Model):
    raise ValueError(
        'Expected model to be a `tf.keras.Model` instance but got: ', model)

  def _strip_clustering_wrapper(layer):
    if isinstance(layer, tf.keras.Model):
      return tf.keras.models.clone_model(
          layer, input_tensors=None, clone_function=_strip_clustering_wrapper)

    elif isinstance(layer, cluster_wrapper.ClusterWeightsMHA):
      # Update cluster associations in order to get the latest weights
      layer.update_clustered_weights_associations()

      # In case of MHA layer, use the overloaded implementation
      return layer.strip_clustering()

    elif isinstance(layer, cluster_wrapper.ClusterWeights):
      # Update cluster associations in order to get the latest weights
      layer.update_clustered_weights_associations()

      # Construct a list of weights to initialize the clean layer
      # non-clusterable weights only
      updated_weights = layer.layer.get_weights()
      for (position_variable,
           weight_name) in layer.position_original_weights.items():
        # Add the clustered weights at the correct position
        clustered_weight = layer.get_weight_from_layer(weight_name)
        updated_weights.insert(position_variable, clustered_weight)

      # Construct a clean layer with the updated weights
      clean_layer = layer.layer.from_config(layer.layer.get_config())
      clean_layer.build(layer.build_input_shape)
      clean_layer.set_weights(updated_weights)

      return clean_layer

    return layer

  # Just copy the model with the right callback
  return tf.keras.models.clone_model(
      model, input_tensors=None, clone_function=_strip_clustering_wrapper)


def layer_has_enough_weights_to_cluster(layer, number_of_clusters,
                                        cluster_per_channel):
  """Returns whether layer has enough weights to cluster.

  Returns True if Conv2D layer has sufficient number of
  weights to implement clustering, given an input number of clusters.

  Args:
    layer: input layer to return quantize configs for.
    number_of_clusters: A number of cluster centroids to form clusters.
    cluster_per_channel: An optional boolean value.
  """
  if not isinstance(layer, tf.keras.layers.Conv2D):
    raise ValueError(f'Input layer should be Conv2D layer: {layer.name} given.')

  if not layer.trainable_weights:
    raise ValueError(f'Layer {layer.name} has no weights to cluster.')

  number_of_layer_weights = tf.cast(tf.size(getattr(layer, 'kernel')), tf.int32)
  channel_idx = 1 if layer.data_format == 'channels_first' else -1
  number_of_channels = tf.size(layer.trainable_weights[channel_idx])

  if cluster_per_channel:
    weights_to_cluster = number_of_layer_weights / number_of_channels
  else:
    weights_to_cluster = number_of_layer_weights

  if weights_to_cluster <= number_of_clusters:
    has_enough_weights = False
  else:
    has_enough_weights = True

  if not has_enough_weights:
    warnings.warn(
        f"Layer {layer.name} does not have enough weights to implement"
        f"{'per-channel ' if cluster_per_channel else ''}clustering."
        f" \nNo clustering was implemented for this layer.\n")
  return has_enough_weights
