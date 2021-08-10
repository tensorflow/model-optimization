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
import distutils.version

import tensorflow as tf

from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper
from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids
from tensorflow_model_optimization.python.core.clustering.keras.clustering_registry import ClusteringRegistry

k = tf.keras.backend
CustomObjectScope = tf.keras.utils.CustomObjectScope
CentroidInitialization = cluster_config.CentroidInitialization
Layer = tf.keras.layers.Layer
InputLayer = tf.keras.layers.InputLayer

# After tf version 2.4.0 the internal variable
# _layers has been renamed to _self_tracked_trackables.
# This variable is the only way to add cluster wrapper
# to layers of a subclassed model.
TF_VERSION_LAYERS = "2.4.0"

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

def _type_model(model):
  """ Auxiliary function to check type of the model:
    Sequential/Functional, Layer or Subclassed.

  Args:
      model : provided model to check

  Returns:
      [tuple]: (is_sequential_or_functional, is_keras_layer, is_subclassed_model)
  """
  is_sequential_or_functional = isinstance(
      model, tf.keras.Model) and (isinstance(model, tf.keras.Sequential) or
                                  model._is_graph_network)

  is_keras_layer = isinstance(
      model, tf.keras.layers.Layer) and not isinstance(model, tf.keras.Model)

  is_subclassed_model = isinstance(model, tf.keras.Model) and \
    not model._is_graph_network

  return (is_sequential_or_functional, is_keras_layer, is_subclassed_model)

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
  (subclass of `tf.keras.layers.Layer`), list of keras layers or a keras model
  (instance of `tf.keras.models.Model`) and handles them appropriately.

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
  (subclass of `tf.keras.layers.Layer`), list of keras layers or a keras model
  (instance of `tf.keras.models.Model`) and handles them appropriately.

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

    return cluster_wrapper.ClusterWeights(layer, number_of_clusters,
                                          cluster_centroids_init,
                                          preserve_sparsity, **kwargs)

  def _wrap_list(layers):
    output = []
    for layer in layers:
      output.append(_add_clustering_wrapper(layer))

    return output

  (is_sequential_or_functional, is_keras_layer, is_subclassed_model) =\
    _type_model(to_cluster)

  if isinstance(to_cluster, list):
    return _wrap_list(to_cluster)
  elif is_sequential_or_functional:
    return tf.keras.models.clone_model(to_cluster,
                                    input_tensors=None,
                                    clone_function=_add_clustering_wrapper)
  elif is_keras_layer:
    return _add_clustering_wrapper(layer=to_cluster)
  elif is_subclassed_model:
    # If the subclassed model is provided, then
    # we add wrappers for all available layers and
    # we wrap the whole model, so that augmented
    # 'build' and 'call' functions are called.

    tf_version = distutils.version.LooseVersion(tf.__version__)
    layers_tf_version = distutils.version.LooseVersion(TF_VERSION_LAYERS)
    for i, layer in enumerate(to_cluster.submodules):
      if tf_version > layers_tf_version:
        to_cluster._self_tracked_trackables[i] = _add_clustering_wrapper(layer=layer)
      else:
        to_cluster._layers[i] = _add_clustering_wrapper(layer=layer)
    return cluster_wrapper.WrapperSubclassedModel(to_cluster)
  else:
    raise ValueError(
        'Clustering cannot be applied. You passed '
        'an object of type: {input}. It should be a keras model or a layer '
        'or a list of layers'.format(input=to_cluster.__class__.__name__))

def strip_clustering(to_strip):
  """Strips clustering wrappers from the model.

  Once a model has been clustered, this method can be used
  to restore the original model or layer with the clustered weights.

  Sequential, functional and subclassed models are supported.

  Arguments:
      to_strip: A `tf.keras.Model` instance with clustered layers or a
        `tf.keras.layers.Layer` instance

  Returns:
    A keras model or layer with clustering wrappers removed.

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
  if not isinstance(to_strip, tf.keras.Model) and not isinstance(
      to_strip, tf.keras.layers.Layer):
    raise ValueError(
        'Expected to_strip to be a `tf.keras.Model` or \
           `tf.keras.layers.Layer` instance but got: ', to_strip)

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

  (is_sequential_or_functional, is_keras_layer, is_subclassed_model) =\
    _type_model(to_strip)

  # Just copy the model with the right callback
  if is_sequential_or_functional:
    return tf.keras.models.clone_model(to_strip,
                                  input_tensors=None,
                                  clone_function=_strip_clustering_wrapper)
  elif is_keras_layer:
    if isinstance(to_strip, tf.keras.layers.Layer):
      return _strip_clustering_wrapper(to_strip)
  elif is_subclassed_model:
    to_strip_model = to_strip.model
    tf_version = distutils.version.LooseVersion(tf.__version__)
    layers_tf_version = distutils.version.LooseVersion(TF_VERSION_LAYERS)
    if tf_version > layers_tf_version:
      for i, layer in enumerate(to_strip_model._self_tracked_trackables):
        to_strip_model._self_tracked_trackables[i] = _strip_clustering_wrapper(layer=layer)
    else:
      for i, layer in enumerate(to_strip_model._layers):
        to_strip_model._layers[i] = _strip_clustering_wrapper(layer=layer)
    return to_strip_model
  else:
    raise ValueError(
        ' Strip clustering cannot be applied. You passed '
        'an object of type: {input}.'.format(input=to_strip.__class__.__name__))

