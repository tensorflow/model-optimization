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

import operator

import tensorflow as tf

from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras import clusterable_layer
from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids
from tensorflow_model_optimization.python.core.clustering.keras import clustering_registry

attrgetter = operator.attrgetter  # pylint: disable=invalid-name
keras = tf.keras
k = keras.backend
Layer = keras.layers.Layer
Wrapper = keras.layers.Wrapper
CentroidInitialization = cluster_config.CentroidInitialization
GradientAggregation = cluster_config.GradientAggregation


class ClusterWeights(Wrapper):
  """This wrapper augments a layer so the weight tensor(s) can be clustered.

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
               cluster_centroids_init=CentroidInitialization.KMEANS_PLUS_PLUS,
               preserve_sparsity=False,
               cluster_per_channel=False,
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
          **kwargs)
    else:
      raise ValueError(
          'Please initialize `Cluster` with a supported layer. Layers should '
          'either be a `ClusterableLayer` instance, or should be supported by '
          'the ClusteringRegistry. You passed: {input}'.format(
              input=layer.__class__))

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

    # Whether to cluster Conv2D kernels per-channel.
    # In case the layer isn't a Conv2D, this isn't applicable
    self.cluster_per_channel = (
        cluster_per_channel if isinstance(layer, tf.keras.layers.Conv2D)
        else False)

    # Number of channels in a Conv2D layer, to be used the case of per-channel
    # clustering.
    self.num_channels = None

    # Whether to apply sparsity preservation or not
    self.preserve_sparsity = preserve_sparsity

    # The way to aggregate the gradient of each cluster centroid
    self.cluster_gradient_aggregation = cluster_gradient_aggregation

    # Stores the pairs of weight names and their respective sparsity masks
    self.sparsity_masks = {}

    # Stores the pairs of weight names and the zero centroids
    self.zero_idx = {}

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
    if (not hasattr(self, '_batch_input_shape') and
        hasattr(layer, '_batch_input_shape')):
      self._batch_input_shape = self.layer._batch_input_shape

    # In the case of Conv2D layer, the data_format needs to be preserved to be
    # used for per-channel clustering
    if hasattr(layer, 'data_format'):
      self.data_format = self.layer.data_format
    else:
      self.data_format = 'channels_last'

    # Save the input shape specified in the build
    self.build_input_shape = None

  def _make_layer_name(self, layer):
    return '{}_{}'.format('cluster', layer.name)

  def _get_zero_idx_mask(self, centroids, zero_cluster):
    zero_idx_mask = (tf.cast(tf.math.not_equal(centroids,
                                               zero_cluster),
                             dtype=tf.float32))
    return zero_idx_mask

  def _get_zero_centroid(self, centroids, zero_idx_mask):
    zero_centroid = tf.math.multiply(centroids,
                                     zero_idx_mask)
    return zero_centroid

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
      # The actual weight_name here for the clustering wrapper is not
      # necessarily the same as the original one from the layer wrapped.
      # For example for cells in StackedRNNCell, the names become
      # 'kernel/0', 'recurrent_kernel/0', 'kernel/1', 'recurrent_kernel/1'
      original_weight = self.get_weight_from_layer(weight_name)
      self.original_clusterable_weights[weight_name] = original_weight
      # Track the variable
      setattr(self, 'original_weight_' + weight_name,
              original_weight)
      # Store the position in layer.weights of original_weight to restore during
      # stripping
      position_original_weight = next(
          i for i, w in enumerate(self.layer.weights) if w is original_weight)
      self.position_original_weights[position_original_weight] = weight_name

      # In the case of per-channel clustering, the number of channels,
      # per-channel number of clusters, as well as the overall number
      # of clusters all need to be preserved in the wrapper.
      if self.cluster_per_channel:
        self.num_channels = (
            original_weight.shape[1] if self.data_format == 'channels_first'
            else original_weight.shape[-1])

      centroid_init_factory = clustering_centroids.CentroidsInitializerFactory
      centroid_init = centroid_init_factory.get_centroid_initializer(
          self.cluster_centroids_init)(weight, self.number_of_clusters,
                                       self.cluster_per_channel,
                                       self.data_format,
                                       self.preserve_sparsity)

      # Init the cluster centroids
      cluster_centroids = (centroid_init.get_cluster_centroids())

      self.cluster_centroids[weight_name] = self.add_weight(
          '{}{}'.format('cluster_centroids_', weight_name),
          shape=(cluster_centroids.shape),
          dtype=weight.dtype,
          trainable=True,
          initializer=tf.keras.initializers.Constant(value=cluster_centroids))

      # Init the weight clustering algorithm
      if isinstance(self.layer, tf.keras.layers.RNN):
        if isinstance(self.layer.cell, tf.keras.layers.StackedRNNCells):
          weight_name_no_index = weight_name.split('/')[0]
        else:
          weight_name_no_index = weight_name
      elif isinstance(self.layer, tf.keras.layers.Bidirectional):
        weight_name_no_index = weight_name.split('/')[0]
      else:
        weight_name_no_index = weight_name
      self.clustering_algorithms[weight_name] = (
          clustering_registry.ClusteringLookupRegistry().get_clustering_impl(
              self.layer, weight_name_no_index, self.cluster_per_channel)
          (
              clusters_centroids=self.cluster_centroids[weight_name],
              cluster_gradient_aggregation=self.cluster_gradient_aggregation,
              data_format=self.data_format,
          ))

      # Init the pulling_indices (weights associations)
      pulling_indices = (
          self.clustering_algorithms[weight_name].get_pulling_indices(
              weight))
      self.pulling_indices[weight_name] = self.add_weight(
          '{}{}'.format('pulling_indices_', weight_name),
          shape=pulling_indices.shape,
          dtype=tf.int64,
          trainable=False,
          synchronization=tf.VariableSynchronization.ON_READ,
          aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
          initializer=tf.keras.initializers.Constant(value=pulling_indices))

      if self.preserve_sparsity:
        # Init the sparsity mask
        clustered_weights = (
            self.clustering_algorithms[weight_name].get_clustered_weight(
                pulling_indices, original_weight))
        self.sparsity_masks[weight_name] = (
            tf.cast(tf.math.not_equal(clustered_weights, 0), dtype=tf.float32))
        # If the model is pruned (which we suppose), this is approximately zero
        self.zero_idx[weight_name] = tf.argmin(
            tf.abs(self.cluster_centroids[weight_name]), axis=-1)

  def update_clustered_weights_associations(self):
    for weight_name, original_weight in self.original_clusterable_weights.items(
    ):

      if self.preserve_sparsity:
        # In the case of per-channel clustering, sparsity
        # needs to be preserved per-channel
        if self.cluster_per_channel:
          for channel in range(self.num_channels):
            zero_idx_mask = (
                self._get_zero_idx_mask(
                    self.cluster_centroids[weight_name][channel],
                    self.cluster_centroids[weight_name][channel][
                        self.zero_idx[weight_name][channel]]))
            self.cluster_centroids[weight_name][channel].assign(
                self._get_zero_centroid(
                    self.cluster_centroids[weight_name][channel],
                    zero_idx_mask))
        else:
          # Set the smallest centroid to zero to force sparsity
          # and avoid extra cluster from forming
          zero_idx_mask = self._get_zero_idx_mask(
              self.cluster_centroids[weight_name],
              self.cluster_centroids[weight_name][self.zero_idx[weight_name]])
          self.cluster_centroids[weight_name].assign(
              self._get_zero_centroid(self.cluster_centroids[weight_name],
                                      zero_idx_mask))

        # During training, the original zero weights can drift slightly.
        # We want to prevent this by forcing them to stay zero at the places
        # where they were originally zero to begin with.
        original_weight = tf.math.multiply(original_weight,
                                           self.sparsity_masks[weight_name])

      # Update pulling indices (cluster associations)
      pulling_indices = (
          self.clustering_algorithms[weight_name].get_pulling_indices(
              original_weight))
      self.pulling_indices[weight_name].assign(pulling_indices)

      # Update clustered weights
      clustered_weights = (
          self.clustering_algorithms[weight_name].get_clustered_weight(
              pulling_indices, original_weight))

      # Replace the weights with their clustered counterparts
      # Remove weight_name index so the wrapper layer weight_name can match
      # the original one
      self.set_weight_to_layer(weight_name,
                               clustered_weights)

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
        'cluster_per_channel': self.cluster_per_channel,
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
    cluster_per_channel = config.pop('cluster_per_channel')

    config['number_of_clusters'] = number_of_clusters
    config['cluster_centroids_init'] = cluster_config.CentroidInitialization(
        cluster_centroids_init)
    config['preserve_sparsity'] = preserve_sparsity
    config['cluster_gradient_aggregation'] = cluster_gradient_aggregation
    config['cluster_per_channel'] = cluster_per_channel

    layer = tf.keras.layers.deserialize(
        config.pop('layer'), custom_objects=custom_objects)
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


class ClusterWeightsRNN(ClusterWeights):
  """This wrapper augments a RNN layer so that the weights can be clustered.

  The weight_name of a single cell in RNN layers is marked with an index in
  registry. In the wrapper layer, the index needs to be removed for matching
  the attribute of the cell layer.
  """

  def get_weight_name_without_index(self, weight_name):
    weight_name_with_index = weight_name.split('/')
    return weight_name_with_index[0], int(weight_name_with_index[1])

  def get_return_layer_cell(self, index):
    return_layer_cell = (self.layer.forward_layer.cell if index == 0 else
                         self.layer.backward_layer.cell)
    return return_layer_cell

  def get_weight_from_layer(self, weight_name):
    weight_name_no_index, i = self.get_weight_name_without_index(weight_name)
    if hasattr(self.layer, 'cell'):
      if isinstance(self.layer.cell, tf.keras.layers.StackedRNNCells):
        return getattr(self.layer.cell.cells[i], weight_name_no_index)
      else:
        return getattr(self.layer.cell, weight_name_no_index)
    elif isinstance(self.layer, tf.keras.layers.Bidirectional):
      if i < 0 or i > 1:
        raise ValueError(
            'Unsupported number of cells in the layer to get weights from.')
      return_layer_cell = self.get_return_layer_cell(i)
      return getattr(return_layer_cell, weight_name_no_index)
    else:
      raise ValueError('No cells in the RNN layer to get weights from.')

  def set_weight_to_layer(self, weight_name, new_weight):
    weight_name_no_index, i = self.get_weight_name_without_index(weight_name)
    if hasattr(self.layer, 'cell'):
      if isinstance(self.layer.cell, tf.keras.layers.StackedRNNCells):
        return setattr(self.layer.cell.cells[i],
                       weight_name_no_index,
                       new_weight)
      else:
        return setattr(self.layer.cell, weight_name_no_index, new_weight)
    elif isinstance(self.layer, tf.keras.layers.Bidirectional):
      if i < 0 or i > 1:
        raise ValueError(
            'Unsupported number of cells in the layer to set weights for.')
      return_layer_cell = self.get_return_layer_cell(i)
      return setattr(return_layer_cell, weight_name_no_index, new_weight)
    else:
      raise ValueError('No cells in the RNN layer to set weights for.')


class ClusterWeightsMHA(ClusterWeights):
  """This wrapper augments a keras MHA layer so that the weights can be clustered."""

  def get_weight_from_layer(self, weight_name):
    pre, _, post = weight_name.rpartition('.')
    return getattr(getattr(self.layer, pre), post)

  def set_weight_to_layer(self, weight_name, new_weight):
    pre, _, post = weight_name.rpartition('.')
    layer = attrgetter(pre)(self.layer)
    setattr(layer, post, new_weight)

  def strip_clustering(self):
    """The restore from config is not working for MHA layer.

    Weights are not created when the build function is called. Therefore,
    original weights have been replaced in the layer.

    Returns:
      Updated layer.
    """
    for weight_name, original_weight in (
        self.original_clusterable_weights.items()):

      # Get the clustered weights
      clustered_weight = self.get_weight_from_layer(weight_name)

      # Re-assign these weights to the original
      original_weight.assign(clustered_weight)
      setattr(self.layer, weight_name, original_weight)

    return self.layer

