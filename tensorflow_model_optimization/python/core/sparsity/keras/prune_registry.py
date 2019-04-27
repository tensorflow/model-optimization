# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.keras import layers

from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer


class PruneRegistry(object):
  """Registry responsible for built-in keras layers."""

  # The keys represent built-in keras layers and the values represent the
  # the variables within the layers which hold the kernel weights. This
  # allows the wrapper to access and modify the weights.
  _LAYERS_WEIGHTS_MAP = {
      layers.advanced_activations.ELU: [],
      layers.advanced_activations.LeakyReLU: [],
      layers.advanced_activations.ReLU: [],
      layers.advanced_activations.Softmax: [],
      layers.advanced_activations.ThresholdedReLU: [],
      layers.convolutional.Conv1D: ['kernel'],
      layers.convolutional.Conv2D: ['kernel'],
      layers.convolutional.Conv2DTranspose: ['kernel'],
      layers.convolutional.Conv3D: ['kernel'],
      layers.convolutional.Conv3DTranspose: ['kernel'],
      layers.convolutional.Cropping1D: [],
      layers.convolutional.Cropping2D: [],
      layers.convolutional.Cropping3D: [],
      layers.convolutional.DepthwiseConv2D: [],
      layers.convolutional.SeparableConv1D: ['pointwise_kernel'],
      layers.convolutional.SeparableConv2D: ['pointwise_kernel'],
      layers.convolutional.UpSampling1D: [],
      layers.convolutional.UpSampling2D: [],
      layers.convolutional.UpSampling3D: [],
      layers.convolutional.ZeroPadding1D: [],
      layers.convolutional.ZeroPadding2D: [],
      layers.convolutional.ZeroPadding3D: [],
      layers.core.Activation: [],
      layers.core.ActivityRegularization: [],
      layers.core.Dense: ['kernel'],
      layers.core.Dropout: [],
      layers.core.Flatten: [],
      layers.core.Lambda: [],
      layers.core.Masking: [],
      layers.core.Permute: [],
      layers.core.RepeatVector: [],
      layers.core.Reshape: [],
      layers.core.SpatialDropout1D: [],
      layers.core.SpatialDropout2D: [],
      layers.core.SpatialDropout3D: [],
      layers.embeddings.Embedding: ['embeddings'],
      layers.local.LocallyConnected1D: ['kernel'],
      layers.local.LocallyConnected2D: ['kernel'],
      layers.merge.Add: [],
      layers.merge.Average: [],
      layers.merge.Concatenate: [],
      layers.merge.Dot: [],
      layers.merge.Maximum: [],
      layers.merge.Minimum: [],
      layers.merge.Multiply: [],
      layers.merge.Subtract: [],
      layers.noise.AlphaDropout: [],
      layers.noise.GaussianDropout: [],
      layers.noise.GaussianNoise: [],
      layers.normalization.BatchNormalization: [],
      layers.normalization.LayerNormalization: [],
      layers.pooling.AveragePooling1D: [],
      layers.pooling.AveragePooling2D: [],
      layers.pooling.AveragePooling3D: [],
      layers.pooling.GlobalAveragePooling1D: [],
      layers.pooling.GlobalAveragePooling2D: [],
      layers.pooling.GlobalAveragePooling3D: [],
      layers.pooling.GlobalMaxPooling1D: [],
      layers.pooling.GlobalMaxPooling2D: [],
      layers.pooling.GlobalMaxPooling3D: [],
      layers.pooling.MaxPooling1D: [],
      layers.pooling.MaxPooling2D: [],
      layers.pooling.MaxPooling3D: [],
  }

  _RNN_CELLS_WEIGHTS_MAP = {
      layers.recurrent.GRUCell: ['kernel', 'recurrent_kernel'],
      layers.recurrent.LSTMCell: ['kernel', 'recurrent_kernel'],
      layers.recurrent.PeepholeLSTMCell: ['kernel', 'recurrent_kernel'],
      layers.recurrent.SimpleRNNCell: ['kernel', 'recurrent_kernel'],
  }

  _RNN_LAYERS = {
      layers.recurrent.GRU,
      layers.recurrent.LSTM,
      layers.recurrent.RNN,
      layers.recurrent.SimpleRNN,
  }

  _RNN_CELLS_STR = ', '.join(str(_RNN_CELLS_WEIGHTS_MAP.keys()))

  _RNN_CELL_ERROR_MSG = (
      'RNN Layer {} contains cell type {} which is either not supported or does'
      'not inherit PrunableLayer. The cell must be one of {}, or implement '
      'PrunableLayer.')

  @classmethod
  def supports(cls, layer):
    """Returns whether the registry supports this layer type.

    Args:
      layer: The layer to check for support.

    Returns:
      True/False whether the layer type is supported.

    """
    if layer.__class__ in cls._LAYERS_WEIGHTS_MAP:
      return True

    if layer.__class__ in cls._RNN_LAYERS:
      for cell in cls._get_rnn_cells(layer):
        if cell.__class__ not in cls._RNN_CELLS_WEIGHTS_MAP \
            and not isinstance(cell, prunable_layer.PrunableLayer):
          return False
      return True

    return False

  @staticmethod
  def _get_rnn_cells(rnn_layer):
    if isinstance(rnn_layer.cell, layers.recurrent.StackedRNNCells):
      return rnn_layer.cell.cells
    else:
      return [rnn_layer.cell]

  @classmethod
  def _is_rnn_layer(cls, layer):
    return layer.__class__ in cls._RNN_LAYERS

  @classmethod
  def _weight_names(cls, layer):
    return cls._LAYERS_WEIGHTS_MAP[layer.__class__]

  @classmethod
  def make_prunable(cls, layer):
    """Modifies a built-in layer object to support pruning.

    Args:
      layer: layer to modify for support.

    Returns:
      The modified layer object.

    """

    if not cls.supports(layer):
      raise ValueError('Layer ' + str(layer.__class__) + ' is not supported.')

    def get_prunable_weights():
      return [getattr(layer, weight) for weight in cls._weight_names(layer)]

    def get_prunable_weights_rnn():  # pylint: disable=missing-docstring
      def get_prunable_weights_rnn_cell(cell):
        if cell.__class__ in cls._RNN_CELLS_WEIGHTS_MAP:
          return [getattr(cell, weight)
                  for weight in cls._RNN_CELLS_WEIGHTS_MAP[cell.__class__]]

        if isinstance(cell, prunable_layer.PrunableLayer):
          return cell.get_prunable_weights()

        raise ValueError(
            cls._RNN_CELL_ERROR_MSG.format(layer.__class__, cell.__class__))

      prunable_weights = []
      for rnn_cell in cls._get_rnn_cells(layer):
        prunable_weights.extend(get_prunable_weights_rnn_cell(rnn_cell))
      return prunable_weights

    if cls._is_rnn_layer(layer):
      layer.get_prunable_weights = get_prunable_weights_rnn
    else:
      layer.get_prunable_weights = get_prunable_weights

    return layer
