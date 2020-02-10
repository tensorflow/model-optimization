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

from tensorflow.python import keras
from tensorflow.python.keras import backend as k
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.utils.generic_utils import custom_object_scope

from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids
from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper


def cluster_scope():
  """Provides a scope in which Clustered layers and models can be deserialized.

  If a keras model or layer has been clustered, it needs to be within this scope
  to be successfully deserialized.

  Returns:
      Object of type `CustomObjectScope` with clustering objects included.

  Example:

  ```python
  clustered_model = cluster_weights(model, **self.params)
  keras.models.save_model(clustered_model, keras_file)

  with cluster_scope():
    loaded_model = keras.models.load_model(keras_file)
  ```
  """
  return custom_object_scope(
      {
          'ClusterWeights': cluster_wrapper.ClusterWeights
      }
  )


def cluster_weights(to_cluster,
                    number_of_clusters,
                    cluster_centroids_init,
                    **kwargs):
  """Modify a keras layer or model to be clustered during training.

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
    'cluster_centroids_init': 'density-based'
  }

  clustered_model = cluster_weights(original_model, **clustering_params)
  ```

  Cluster a layer:

  ```python
  clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': 'density-based'
  }

  model = keras.Sequential([
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
      cluster_centroids_init: how to initialize the cluster centroids.
        Can have following values:
          1. 'random' : centroids are sampled using the uniform distribution
          between the minimum and maximum weight values in a given layer
          2. 'density-based' : density-based sampling. First, cumulative
          distribution function is built for weights, then y-axis is evenly
          spaced into number_of_clusters regions. After this the corresponding x
          values are obtained and used to initialize clusters centroids.
          3. 'linear' : cluster centroids are evenly spaced between the minimum
          and maximum values of a given weight
      **kwargs: Additional keyword arguments to be passed to the keras layer.
        Ignored when to_cluster is not a keras layer.

  Returns:
    Layer or model modified to include clustering related metadata.

  Raises:
    ValueError: if the keras layer is unsupported, or the keras model contains
    an unsupported layer.
  """
  if not clustering_centroids.CentroidsInitializerFactory.\
      init_is_supported(cluster_centroids_init):
    raise ValueError("cluster centroids can only be one of three values: "
                     "random, density-based, linear")

  def _add_clustering_wrapper(layer):
    if isinstance(layer, cluster_wrapper.ClusterWeights):
      return layer
    if isinstance(layer, InputLayer):
      return layer.__class__.from_config(layer.get_config())

    return cluster_wrapper.ClusterWeights(layer,
                                          number_of_clusters,
                                          cluster_centroids_init,
                                          **kwargs)

  def _wrap_list(layers):
    output = []
    for layer in layers:
      output.append(_add_clustering_wrapper(layer))

    return output

  if isinstance(to_cluster, keras.Model):
    return keras.models.clone_model(to_cluster,
                                    input_tensors=None,
                                    clone_function=_add_clustering_wrapper)
  if isinstance(to_cluster, Layer):
    return _add_clustering_wrapper(layer=to_cluster)
  if isinstance(to_cluster, list):
    return _wrap_list(to_cluster)


def strip_clustering(model):
  """Strip clustering wrappers from the model.

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
  if not isinstance(model, keras.Model):
    raise ValueError(
        'Expected model to be a `tf.keras.Model` instance but got: ', model)

  def _strip_clustering_wrapper(layer):
    if isinstance(layer, cluster_wrapper.ClusterWeights):
      if not hasattr(layer.layer, '_batch_input_shape') and\
          hasattr(layer, '_batch_input_shape'):
        layer.layer._batch_input_shape = layer._batch_input_shape

      # We reset both arrays of weights, so that we can guarantee the correct
      # order of newly created weights
      layer.layer._trainable_weights = []
      layer.layer._non_trainable_weights = []
      for i in range(len(layer.restore)):
        # This is why we used integers as keys
        name, weight = layer.restore[i]
        # In both cases we use k.batch_get_value since we need physical copies
        # of the arrays to initialize a new tensor
        if i in layer.gone_variables:
          # If the variable was removed because it was clustered, we restore it
          # by using updater we created earlier
          new_weight_value = k.batch_get_value([weight()])[0]
        else:
          # If the value was not clustered(e.g. bias), we still store a valid
          # reference to the tensor. We use this reference to get the value
          new_weight_value = k.batch_get_value([weight])[0]
        layer.layer.add_weight(
            name=name,
            shape=new_weight_value.shape,
            initializer=initializers.Constant(new_weight_value),
            trainable=True
        )
      # When all weights are filled with the values, just return the underlying
      # layer since it is now fully autonomous from its wrapper
      return layer.layer
    return layer

  # Just copy the model with the right callback
  return keras.models.clone_model(model,
                                  input_tensors=None,
                                  clone_function=_strip_clustering_wrapper)
