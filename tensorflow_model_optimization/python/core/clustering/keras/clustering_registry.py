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

from tensorflow_model_optimization.python.core.clustering.keras import clusterable_layer
from tensorflow_model_optimization.python.core.clustering.keras import clustering_algorithm

ClusteringAlgorithm = clustering_algorithm.ClusteringAlgorithm

class ClusteringLookupRegistry(object):

  @classmethod
  def get_clustering_impl(cls, layer, weight_name):
    """Returns a certain reshape/lookup implementation for a given array.

    Args:
      layer: A layer that is being clustered
      weight_name: concrete weight name to be clustered.
    Returns:
      A concrete implementation of a lookup algorithm.
    """

    # Clusterable layer could provide own implementation of get_pulling_indices
    if (issubclass(layer.__class__, clusterable_layer.ClusterableLayer) and
      layer.get_clusterable_algorithm is not None):
      ans = layer.get_clusterable_algorithm(weight_name)
      if ans:
        return ans
    return ClusteringAlgorithm


class ClusteringRegistry(object):
  """Registry responsible for built-in keras layers."""

  # The keys represent built-in keras layers and the values represent the
  # the variables within the layers which hold the kernel weights. This
  # allows the wrapper to access and modify the weights.
  _LAYERS_WEIGHTS_MAP = {
      tf.keras.layers.Conv1D: ['kernel'],
      tf.keras.layers.Conv2D: ['kernel'],
      tf.keras.layers.Conv2DTranspose: ['kernel'],
      tf.keras.layers.Conv3D: ['kernel'],
      tf.keras.layers.Conv3DTranspose: ['kernel'],
      # non-clusterable due to big unrecoverable accuracy loss
      tf.keras.layers.DepthwiseConv2D: [],
      tf.keras.layers.SeparableConv1D: ['pointwise_kernel'],
      tf.keras.layers.SeparableConv2D: ['pointwise_kernel'],
      tf.keras.layers.Dense: ['kernel'],
      tf.keras.layers.Embedding: ['embeddings'],
      tf.keras.layers.LocallyConnected1D: ['kernel'],
      tf.keras.layers.LocallyConnected2D: ['kernel'],
      tf.keras.layers.BatchNormalization: [],
      tf.keras.layers.LayerNormalization: [],
  }

  _SUPPORTED_RNN_LAYERS = {
      tf.keras.layers.GRU,
      tf.keras.layers.GRUCell,
      tf.keras.layers.LSTM,
      tf.keras.layers.LSTMCell,
      tf.keras.layers.SimpleRNN,
      tf.keras.layers.SimpleRNNCell,
      tf.keras.layers.StackedRNNCells,
  }

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
    if not layer.trainable_weights and not isinstance(
        layer, tf.keras.layers.RNN):
      return True

    if layer.__class__ in cls._LAYERS_WEIGHTS_MAP:
      return True

    if hasattr(layer, 'cell'):
      if layer.cell.__class__ in cls._SUPPORTED_RNN_LAYERS:
        return True

    if layer.__class__ in cls._SUPPORTED_RNN_LAYERS:
      return True

    return False

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
      return [(weight_name, getattr(layer, weight_name))
              for weight_name in cls._weight_names(layer)]

    def get_clusterable_weights_rnn():  # pylint: disable=missing-docstring
      if isinstance(layer.cell, clusterable_layer.ClusterableLayer):
        raise ValueError(
            "ClusterableLayer is not yet supported for RNNs based layer.")
      else:
        clusterable_weights = [
          ('kernel', layer.cell.kernel),
          ('recurrent_kernel', layer.cell.recurrent_kernel),
        ]
        return clusterable_weights

    def get_clusterable_weights_rnn_cells():
      if isinstance(layer.cell, tf.keras.layers.StackedRNNCells):
        clusterable_weights = []
        clusterable_weights.append(['kernel/' + str(i),
                                    layer.cell.cells[i].kernel])
        clusterable_weights.append(['recurrent_kernel/' + str(i),
                                    layer.cell.cells[i].recurrent_kernel])
        return clusterable_weights

    if layer.__class__ in cls._SUPPORTED_RNN_LAYERS:
      layer.get_clusterable_weights = get_clusterable_weights_rnn
    elif hasattr(layer, 'cell') and layer.cell.__class__ in cls._SUPPORTED_RNN_LAYERS:
      for i in range(0, len(layer.cell.cells)):
        if not cls.supports(layer.cell.cells[i]):
          raise ValueError(
              'Layer cell ' + str(
                  layer.cell.cells[i].__class__) + ' is not supported.')
      layer.get_clusterable_weights = get_clusterable_weights_rnn_cells
    else:
      layer.get_clusterable_weights = get_clusterable_weights

    return layer
