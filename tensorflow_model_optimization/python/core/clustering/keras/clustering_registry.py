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
"""Registry responsible for built-in keras classes."""

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow_model_optimization.python.core.clustering.keras import clusterable_layer
from tensorflow_model_optimization.python.core.clustering.keras import clustering_algorithm

AbstractClusteringAlgorithm = clustering_algorithm.AbstractClusteringAlgorithm


class ConvolutionalWeightsCA(AbstractClusteringAlgorithm):
  """Look-ups for convolutional kernels, e.g. tensors with shape [B,W,H,C]."""

  def get_pulling_indices(self, weight):
    wt_dim = len(weight.shape)
    clst_num = self.cluster_centroids.shape[0]
    tiled_weights = tf.tile(tf.expand_dims(weight, wt_dim), [1 for _ in range(wt_dim)] + [clst_num])

    # Do the ugly reshape to the clustering points
    tiled_cluster_centroids = tf.stack(
        [tf.tile(tf.stack(
            [tf.reshape(self.cluster_centroids, [1 for _ in range(wt_dim-2)] + [clst_num])] *
            weight.shape[-2], axis=wt_dim-2),
                 [i for i in weight.shape[:-2]] + [1, 1])] * weight.shape[-1],
        axis=wt_dim-1)

    # We find the nearest cluster centroids and store them so that ops can build
    # their kernels upon it
    pulling_indices = tf.argmin(
        tf.abs(tiled_weights - tiled_cluster_centroids), axis=wt_dim
    )

    return pulling_indices


class DenseWeightsCA(AbstractClusteringAlgorithm):
  """Dense layers store their weights in 2D tables, i.e. tensor shape [U, D]."""

  def get_pulling_indices(self, weight):
    clst_num = self.cluster_centroids.shape[0]
    tiled_weights = tf.tile(tf.expand_dims(weight, axis=2), [1, 1, clst_num])
    tiled_cluster_centroids = tf.tile(
        tf.reshape(self.cluster_centroids, [1, 1, clst_num]),
        [weight.shape[0], weight.shape[1], 1])

    # We find the nearest cluster centroids and store them so that ops can build
    # their kernels upon it
    pulling_indices = tf.argmin(tf.abs(tiled_weights - tiled_cluster_centroids),
                                axis=2)

    return pulling_indices


class BiasWeightsCA(AbstractClusteringAlgorithm):
  """Biases are stored as tensors of rank 0."""

  def get_pulling_indices(self, weight):
    clst_num = self.cluster_centroids.shape[0]
    tiled_weights = tf.tile(tf.expand_dims(weight, axis=1), [1, clst_num])
    tiled_cluster_centroids = tf.tile(
        tf.reshape(self.cluster_centroids, [1, clst_num]), [weight.shape[0], 1])

    pulling_indices = tf.argmin(tf.abs(tiled_weights - tiled_cluster_centroids),
                                axis=1)

    return pulling_indices


class ClusteringLookupRegistry(object):
  """Map of layers to strategy.

  The keys represent built-in keras layers and the values represent the
  strategy accoding to which clustering will be done.
  If the key is not present in the map, that means that there is nothing to
  work on, or the strategy is not currently supported
  """
  _LAYERS_RESHAPE_MAP = {
      layers.Conv1D: {
          'kernel': ConvolutionalWeightsCA,
          'bias': BiasWeightsCA
      },
      layers.Conv2D: {
          'kernel': ConvolutionalWeightsCA,
          'bias': BiasWeightsCA
      },
      layers.Conv2DTranspose: {
          'kernel': ConvolutionalWeightsCA,
          'bias': BiasWeightsCA
      },
      layers.Conv3D: {
          'kernel': ConvolutionalWeightsCA,
          'bias': BiasWeightsCA
      },
      layers.Conv3DTranspose: {
          'kernel': ConvolutionalWeightsCA,
          'bias': BiasWeightsCA
      },
      layers.SeparableConv1D: {
          'pointwise_kernel': ConvolutionalWeightsCA,
          'bias': BiasWeightsCA
      },
      layers.SeparableConv2D: {
          'pointwise_kernel': ConvolutionalWeightsCA,
          'bias': BiasWeightsCA
      },
      layers.Dense: {
          'kernel': DenseWeightsCA,
          'bias': BiasWeightsCA
      },
      layers.Embedding: {
          'embeddings': DenseWeightsCA,
          'bias': BiasWeightsCA
      },
      layers.LocallyConnected1D: {
          'kernel': ConvolutionalWeightsCA,
          'bias': BiasWeightsCA
      },
      layers.LocallyConnected2D: {
          'kernel': ConvolutionalWeightsCA,
          'bias': BiasWeightsCA
      },
  }

  @classmethod
  def register_new_implementation(cls, new_impl):
    """Registers new implementation.

    For custom user-defined objects define the way how clusterable weights
    are going to be formed. If weights are any of these, 1D,2D or 4D, please
    consider using existing implementations: BiasWeightsCA,
    ConvolutionalWeightsCA and DenseWeightsCA.

    Args:
      new_impl: dictionary. Keys are classes and values are dictionaries.
      The latter have strings as keys and values are classes inherited from
      AbstractClusteringAlgorithm. Normally, the set keys of the latter
      dictionaries should match the set of clusterable weights names for the
      layer.
    Returns:
      None
    """
    if not isinstance(new_impl, dict):
      raise TypeError('new_impl must be a dictionary')
    for k, v in new_impl.items():
      if not isinstance(v, dict):
        raise TypeError(
            'Every value of new_impl must be a dictionary. Item for key {key} '
            'has class {vclass}'.format(key=k, vclass=v))

    cls._LAYERS_RESHAPE_MAP.update(new_impl)

  @classmethod
  def get_clustering_impl(cls, layer, weight_name):
    """Returns a certain reshape/lookup implementation for a given array.

    Args:
      layer: A layer that is being clustered
      weight_name: concrete weight name to be clustered.
    Returns:
      A concrete implementation of a lookup algorithm.
    """
    custom_layer_of_built_layer = None
    if layer.__class__ not in cls._LAYERS_RESHAPE_MAP:
      # Checks whether we have a custom layer derived from built-in keras class.
      for key in cls._LAYERS_RESHAPE_MAP:
        if issubclass(layer.__class__, key):
          custom_layer_of_built_layer = key
      if not custom_layer_of_built_layer:
        # Checks whether we have a customerable layer that provides
        # clusterable algorithm for the given weights.
        if (issubclass(layer.__class__, clusterable_layer.ClusterableLayer) and
            layer.get_clusterable_algorithm is not None):
          ans = layer.get_clusterable_algorithm(weight_name)
          if not ans:
            raise ValueError(
                'Class {given_class} does not provide clustering algorithm'
                'for the weights with the name {weight_name}.'.format(
                    given_class=layer.__class__, weight_name=weight_name))
          else:
            return ans
        else:
          raise ValueError(
              'Class {given_class} has not derived from ClusterableLayer'
              'or the funtion get_pulling_indices is not provided.'.format(
                  given_class=layer.__class__))
    else:
      custom_layer_of_built_layer = layer.__class__
    if weight_name not in cls._LAYERS_RESHAPE_MAP[custom_layer_of_built_layer]:
      raise ValueError(
          "Weight with the name '{given_weight_name}' for class {given_class} "
          'has not been registered in the ClusteringLookupRegistry. Use '
          'ClusteringLookupRegistry.register_new_implementation '
          'to fix this.'.format(
              given_class=layer.__class__, given_weight_name=weight_name))
    # Different weights will have different shapes hence there is double hash
    # map lookup.
    return cls._LAYERS_RESHAPE_MAP[custom_layer_of_built_layer][weight_name]


class ClusteringRegistry(object):
  """Registry responsible for built-in keras layers."""

  # The keys represent built-in keras layers and the values represent the
  # the variables within the layers which hold the kernel weights. This
  # allows the wrapper to access and modify the weights.
  _LAYERS_WEIGHTS_MAP = {
      layers.Conv1D: ['kernel'],
      layers.Conv2D: ['kernel'],
      layers.Conv2DTranspose: ['kernel'],
      layers.Conv3D: ['kernel'],
      layers.Conv3DTranspose: ['kernel'],
      # non-clusterable due to big unrecoverable accuracy loss
      layers.DepthwiseConv2D: [],
      layers.SeparableConv1D: ['pointwise_kernel'],
      layers.SeparableConv2D: ['pointwise_kernel'],
      layers.Dense: ['kernel'],
      layers.Embedding: ['embeddings'],
      layers.LocallyConnected1D: ['kernel'],
      layers.LocallyConnected2D: ['kernel'],
      layers.BatchNormalization: [],
      layers.LayerNormalization: [],
  }

  _RNN_CELLS_WEIGHTS_MAP = {
      # NOTE: RNN cells are added via compat.v1 and compat.v2 to support legacy
      # TensorFlow 2.X behavior where the v2 RNN uses the v1 RNNCell instead of
      # the v2 RNNCell.
      tf.compat.v1.keras.layers.GRUCell: ['kernel', 'recurrent_kernel'],
      tf.compat.v2.keras.layers.GRUCell: ['kernel', 'recurrent_kernel'],
      tf.compat.v1.keras.layers.LSTMCell: ['kernel', 'recurrent_kernel'],
      tf.compat.v2.keras.layers.LSTMCell: ['kernel', 'recurrent_kernel'],
      tf.compat.v1.keras.experimental.PeepholeLSTMCell: [
          'kernel', 'recurrent_kernel'
      ],
      tf.compat.v2.keras.experimental.PeepholeLSTMCell: [
          'kernel', 'recurrent_kernel'
      ],
      tf.compat.v1.keras.layers.SimpleRNNCell: ['kernel', 'recurrent_kernel'],
      tf.compat.v2.keras.layers.SimpleRNNCell: ['kernel', 'recurrent_kernel'],
  }

  _RNN_LAYERS = {
      layers.GRU,
      layers.LSTM,
      layers.RNN,
      layers.SimpleRNN,
  }

  _RNN_CELLS_STR = ', '.join(str(_RNN_CELLS_WEIGHTS_MAP.keys()))

  _RNN_CELL_ERROR_MSG = (
      'RNN Layer {} contains cell type {} which is either not supported or does'
      'not inherit ClusterableLayer. The cell must be one of {}, or implement '
      'ClusterableLayer.'
  )

  @classmethod
  def supports(cls, layer):
    """Returns whether the registry supports this layer type.

    Args:
      layer: The layer to check for support.

    Returns:
      True/False whether the layer type is supported.

    """
    # Automatically enable layers with zero trainable weights.
    # Example: Reshape, AveragePooling2D, Maximum/Minimum, etc.
    if not layer.trainable_weights:
      return True

    if layer.__class__ in cls._LAYERS_WEIGHTS_MAP:
      return True

    if layer.__class__ in cls._RNN_LAYERS:
      for cell in cls._get_rnn_cells(layer):
        if (cell.__class__ not in cls._RNN_CELLS_WEIGHTS_MAP
            and not isinstance(cell, clusterable_layer.ClusterableLayer)):
          return False
      return True

    return False

  @staticmethod
  def _get_rnn_cells(rnn_layer):
    if isinstance(rnn_layer.cell, layers.StackedRNNCells):
      return rnn_layer.cell.cells
    return [rnn_layer.cell]

  @classmethod
  def _is_rnn_layer(cls, layer):
    return layer.__class__ in cls._RNN_LAYERS

  @classmethod
  def _weight_names(cls, layer):
    # For layers with zero trainable weights, like Reshape, Pooling.
    if not layer.trainable_weights:
      return []

    return cls._LAYERS_WEIGHTS_MAP[layer.__class__]

  @classmethod
  def make_clusterable(cls, layer):
    """Modifies a built-in layer object to support clustering.

    Args:
      layer: layer to modify for support.

    Returns:
      The modified layer object.

    """

    if not cls.supports(layer):
      raise ValueError('Layer ' + str(layer.__class__) + ' is not supported.')

    def get_clusterable_weights():
      return [(weight, getattr(layer, weight)) for weight in
              cls._weight_names(layer)]

    def get_clusterable_weights_rnn():  # pylint: disable=missing-docstring
      def get_clusterable_weights_rnn_cell(cell):
        if cell.__class__ in cls._RNN_CELLS_WEIGHTS_MAP:
          return [(weight, getattr(cell, weight))
                  for weight in cls._RNN_CELLS_WEIGHTS_MAP[cell.__class__]]

        if isinstance(cell, clusterable_layer.ClusterableLayer):
          return cell.get_clusterable_weights()

        raise ValueError(cls._RNN_CELL_ERROR_MSG.format(
            layer.__class__, cell.__class__, cls._RNN_CELLS_WEIGHTS_MAP.keys()))

      clusterable_weights = []
      for rnn_cell in cls._get_rnn_cells(layer):
        clusterable_weights.extend(get_clusterable_weights_rnn_cell(rnn_cell))
      return clusterable_weights

    if cls._is_rnn_layer(layer):
      layer.get_clusterable_weights = get_clusterable_weights_rnn
    else:
      layer.get_clusterable_weights = get_clusterable_weights

    return layer
