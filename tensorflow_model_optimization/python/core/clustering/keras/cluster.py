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

import tensorflow as tf

from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper
from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids

k = tf.keras.backend
CustomObjectScope = tf.keras.utils.CustomObjectScope
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
  return CustomObjectScope(
      {
          'ClusterWeights': cluster_wrapper.ClusterWeights
      }
  )


def cluster_weights(to_cluster,
                    number_of_clusters,
                    cluster_centroids_init,
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

  Args:
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
  return _cluster_weights(
      to_cluster,
      number_of_clusters,
      cluster_centroids_init,
      preserve_sparsity=False,
      **kwargs)


def _cluster_weights(to_cluster, number_of_clusters, cluster_centroids_init,
                     preserve_sparsity, **kwargs):
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
    'preserve_sparsity': False
  }

  clustered_model = cluster_weights(original_model, **clustering_params)
  ```

  Cluster a layer:

  ```python
  clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': CentroidInitialization.DENSITY_BASED,
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

  Args:
      to_cluster: A single keras layer, list of keras layers, or a
        `tf.keras.Model` instance.
      number_of_clusters: the number of cluster centroids to form when
        clustering a layer/model. For example, if number_of_clusters=8 then only
        8 unique values will be used in each weight array.
      cluster_centroids_init: `tfmot.clustering.keras.CentroidInitialization`
        instance that determines how the cluster centroids will be initialized.
      preserve_sparsity (experimental): optional boolean value that determines
        whether or not sparsity preservation will be enforced during training.
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
    if isinstance(layer, tf.keras.layers.RNN):
      return cluster_wrapper.ClusterWeightsRNN(
          layer,
          number_of_clusters,
          cluster_centroids_init,
          preserve_sparsity,
          **kwargs,
      )

    return cluster_wrapper.ClusterWeights(layer, number_of_clusters,
                                          cluster_centroids_init,
                                          preserve_sparsity, **kwargs)

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

  Args:
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
