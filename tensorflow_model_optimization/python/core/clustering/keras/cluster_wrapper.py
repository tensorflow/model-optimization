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
"""Keras ClusterWeights wrapper API."""

import tensorflow as tf

from tensorflow.keras import initializers

from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras import clusterable_layer
from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids
from tensorflow_model_optimization.python.core.clustering.keras import clustering_registry

keras = tf.keras
k = keras.backend
Layer = keras.layers.Layer
Wrapper = keras.layers.Wrapper
CentroidInitialization = cluster_config.CentroidInitialization
GradientAggregation = cluster_config.GradientAggregation


class ClusterWeights(Wrapper):
  """This wrapper augments a keras layer so that the weight tensor(s) can be clustered.

  This wrapper implements nearest neighbor clustering algorithm. This algorithm
  ensures that only a specified number of unique values are used in a weight
  tensor. This allows for certain types of hardware to benefit from advanced
  weight compression techniques and the associated reduction in model memory
  footprint and bandwidth.

  From practical standpoint this is implemented using a lookup table to hold the
  cluster centroid values during model training. The weight array is populated
  with 'gather' operation so that during back propagation the gradients can be
  calculated in a normal way. The lookup table is then adjusted using the
  cumulative gradient values for the weights that correspond to the same
  centroid.

  The number of unique values required as well as the way cluster centroids
  are initialized are passed in the wrapper's constructor.

  The initial values of cluster centroids are fine-tuned during the training.
  """

  def __init__(self,
               layer,
               number_of_clusters,
               cluster_centroids_init,
               preserve_sparsity=False,
               cluster_gradient_aggregation=GradientAggregation.SUM,
               **kwargs):
    if not isinstance(layer, Layer):
      raise ValueError(
          'Please initialize `Cluster` layer with a '
          '`Layer` instance. You passed: {input}'.format(input=layer))

    if 'name' not in kwargs:
      kwargs['name'] = self._make_layer_name(layer)

    if isinstance(layer, clusterable_layer.ClusterableLayer):
      # A user-defined custom layer
      super(ClusterWeights, self).__init__(layer, **kwargs)
    elif clustering_registry.ClusteringRegistry.supports(layer):
      super(ClusterWeights, self).__init__(
          clustering_registry.ClusteringRegistry.make_clusterable(layer),
          **kwargs
      )
    else:
      raise ValueError(
          'Please initialize `Cluster` with a supported layer. Layers should '
          'either be a `ClusterableLayer` instance, or should be supported by '
          'the ClusteringRegistry. You passed: {input}'.format(
              input=layer.__class__
          )
      )

    if not isinstance(number_of_clusters, int):
      raise ValueError(
          'number_of_clusters must be an integer. Given: {}'.format(
              number_of_clusters.__class__))

    limit_number_of_clusters = 2 if preserve_sparsity else 1
    if number_of_clusters <= limit_number_of_clusters:
      raise ValueError(
          'number_of_clusters must be greater than {}. Given: {}'.format(
              limit_number_of_clusters, number_of_clusters))

    self._track_trackable(layer, name='layer')

    # The way how cluster centroids will be initialized
    self.cluster_centroids_init = cluster_centroids_init

    # The number of cluster centroids
    self.number_of_clusters = number_of_clusters

    # Whether to apply sparsity preservation or not
    self.preserve_sparsity = preserve_sparsity

    # The way to aggregate the gradient of each cluster centroid
    self.cluster_gradient_aggregation = cluster_gradient_aggregation

    # Stores the pairs of weight names and their respective sparsity masks
    self.sparsity_masks = {}

    # Map weight names to original clusterable weights variables
    # Those weights will still be updated during backpropagation
    self.original_clusterable_weights = {}

    # Map the position of the original weight variable in the
    # child layer to the weight name
    self.position_original_weights = {}

    # Map weight names to corresponding clustering algorithms
    self.clustering_algorithms = {}

    # Map weight names to corresponding indices lookup tables
    self.pulling_indices = {}

    # Map weight names to corresponding cluster centroid variables
    self.cluster_centroids = {}

    # If the input shape was specified, then we need to preserve this
    # information in the layer. If this info is not preserved, then the `built`
    # state will not be preserved between serializations.
    if not hasattr(self, '_batch_input_shape')\
        and hasattr(layer, '_batch_input_shape'):
      self._batch_input_shape = self.layer._batch_input_shape

    # Save the input shape specified in the build
    self.build_input_shape = None

  @staticmethod
  def _make_layer_name(layer):
    return '{}_{}'.format('cluster', layer.name)

  def get_weight_from_layer(self, weight_name):
    return getattr(self.layer, weight_name)

  def set_weight_to_layer(self, weight_name, new_weight):
    setattr(self.layer, weight_name, new_weight)

  def build(self, input_shape):
    super(ClusterWeights, self).build(input_shape)
    self.build_input_shape = input_shape

    # For every clusterable weights, create the clustering logic
    for weight_name, weight in self.layer.get_clusterable_weights():
      # Store the original weight in this wrapper
      # The child reference will be overridden in
      # update_clustered_weights_associations
      original_weight = self.get_weight_from_layer(weight_name)
      self.original_clusterable_weights[weight_name] = original_weight
      # Track the variable
      setattr(self, "original_weight_" + weight_name, original_weight)

      # Store the position in layer.weights of original_weight to restore
      # during stripping
      position_original_weight = next(
        i for i, w in enumerate(self.layer.weights) if w is original_weight
      )
      self.position_original_weights[position_original_weight] = weight_name

      # Init the cluster centroids
      cluster_centroids = (
        clustering_centroids.CentroidsInitializerFactory
        .get_centroid_initializer(self.cluster_centroids_init)(
          weight, self.number_of_clusters, self.preserve_sparsity
        )
        .get_cluster_centroids()
      )
      self.cluster_centroids[weight_name] = self.add_weight(
          '{}{}'.format('cluster_centroids_', weight_name),
          shape=(self.number_of_clusters,),
          dtype=weight.dtype,
          trainable=True,
          initializer=initializers.Constant(value=cluster_centroids)
      )

      # Init the weight clustering algorithm
      self.clustering_algorithms[weight_name] = (
        clustering_registry.ClusteringLookupRegistry()
        .get_clustering_impl(self.layer, weight_name)(
          clusters_centroids=self.cluster_centroids[weight_name],
          cluster_gradient_aggregation=self.cluster_gradient_aggregation,
        )
      )

      # Init the pulling_indices (weights associations)
      pulling_indices = (
        self.clustering_algorithms[weight_name]
        .get_pulling_indices(weight)
      )
      self.pulling_indices[weight_name] = self.add_weight(
          '{}{}'.format('pulling_indices_', weight_name),
          shape=pulling_indices.shape,
          dtype=tf.int64,
          trainable=False,
          synchronization=tf.VariableSynchronization.ON_READ,
          aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
          initializer=initializers.Constant(value=pulling_indices)
      )

      if self.preserve_sparsity:
        # Init the sparsity mask
        clustered_weights = (
          self.clustering_algorithms[weight_name]
          .get_clustered_weight(pulling_indices, original_weight)
        )
        self.sparsity_masks[weight_name] = (
          tf.cast(tf.math.not_equal(clustered_weights, 0), dtype=tf.float32)
        )

  def update_clustered_weights_associations(self):
    for weight_name, original_weight in self.original_clusterable_weights.items():
      # Update pulling indices (cluster associations)
      pulling_indices = (
        self.clustering_algorithms[weight_name]
        .get_pulling_indices(original_weight)
      )
      self.pulling_indices[weight_name].assign(pulling_indices)

      # Update clustered weights
      clustered_weights = (
        self.clustering_algorithms[weight_name]
        .get_clustered_weight(
          pulling_indices, original_weight
        )
      )
      if self.preserve_sparsity:
        # Re-discover the sparsity masks to avoid drifting
        self.sparsity_masks[weight_name] = (
            tf.cast(tf.math.not_equal(clustered_weights, 0), dtype=tf.float32)
        )
        # Apply the sparsity mask to the clustered weights
        clustered_weights = tf.math.multiply(clustered_weights, self.sparsity_masks[weight_name])

      # Replace the weights with their clustered counterparts
      self.set_weight_to_layer(weight_name, clustered_weights)

  def call(self, inputs, training=None, **kwargs):
    # Update cluster associations in order to set the latest weights
    self.update_clustered_weights_associations()

    return self.layer.call(inputs, **kwargs)

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(input_shape)

  def get_config(self):
    base_config = super(ClusterWeights, self).get_config()
    config = {
        'number_of_clusters': self.number_of_clusters,
        'cluster_centroids_init': self.cluster_centroids_init,
        'preserve_sparsity': self.preserve_sparsity,
        'cluster_gradient_aggregation': self.cluster_gradient_aggregation,
        **base_config
    }
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()

    number_of_clusters = config.pop('number_of_clusters')
    cluster_centroids_init = config.pop('cluster_centroids_init')
    preserve_sparsity = config.pop('preserve_sparsity')
    cluster_gradient_aggregation = config.pop('cluster_gradient_aggregation')

    config['number_of_clusters'] = number_of_clusters
    config['cluster_centroids_init'] = cluster_config.CentroidInitialization(
        cluster_centroids_init)
    config['preserve_sparsity'] = preserve_sparsity
    config['cluster_gradient_aggregation'] = cluster_gradient_aggregation

    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    layer = deserialize_layer(config.pop('layer'),
                              custom_objects=custom_objects)
    config['layer'] = layer

    return cls(**config)

  @property
  def trainable(self):
    return self.layer.trainable

  @trainable.setter
  def trainable(self, value):
    self.layer.trainable = value

  @property
  def trainable_weights(self):
    return self.layer.trainable_weights + self._trainable_weights

  @property
  def non_trainable_weights(self):
    return self.layer.non_trainable_weights + self._non_trainable_weights

  @property
  def updates(self):
    return self.layer.updates + self._updates

  @property
  def losses(self):
    return self.layer.losses + self._losses

  def get_weights(self):
    return self.layer.get_weights()

  def set_weights(self, weights):
    self.layer.set_weights(weights)

class WrapperSubclassedModel(keras.Model):
  """This wrapper wraps a keras subclassed model so that the weight tensor(s)
  in keras layers that are defined in this model can be clustered.
  """
  def __init__(self, model):
    super(WrapperSubclassedModel, self).__init__()

    # This wrapper is needed only for subclassed models.
    is_subclassed_model = isinstance(model, keras.Model) and \
      not model._is_graph_network
    if not is_subclassed_model:
      raise ValueError(
          "The provided model should be subclassed. The provided: {}".format(
              model.__class__
          )
      )
    self.model = model

  def build(self, input_shape):
    for layer in self.model.layers:
      if isinstance(layer, ClusterWeights):
        layer.build(input_shape = input_shape)
    return self.model.build(input_shape = input_shape)

  def call(self, inputs):
    for layer in self.model.layers:
      if isinstance(layer, ClusterWeights):
        layer.call(inputs)
    return self.model.call(inputs)


class ClusterWeightsRNN(ClusterWeights):
  """This wrapper augments a keras RNN layer so that the weights can be clustered."""

  def get_weight_from_layer(self, weight_name):
    return getattr(self.layer.cell, weight_name)

  def set_weight_to_layer(self, weight_name, new_weight):
    setattr(self.layer.cell, weight_name, new_weight)
