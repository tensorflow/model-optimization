# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Default ternarization QuantizeConfigs."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
from tensorflow_model_optimization.python.core.quantization.keras.experimental.ternarization import ternarization_quantizers

QuantizeConfig = quantize_config.QuantizeConfig

layers = tf.keras.layers


class _RNNHelper(object):
  """Helper functions for working with RNNs."""

  def _get_rnn_cells(self, rnn_layer):
    """Returns the list of cells in an RNN layer."""
    if isinstance(rnn_layer.cell, layers.StackedRNNCells):
      return rnn_layer.cell.cells
    else:
      return [rnn_layer.cell]


class TernarizationQuantizeConfig(QuantizeConfig):
  """QuantizeConfig for non recurrent Keras layers."""

  def __init__(self, weight_attrs):
    self.weight_attrs = weight_attrs

    # TODO(pulkitb): For some layers such as Conv2D, per_axis should be True.
    # Add mapping for which layers support per_axis.
    self.weight_quantizer = ternarization_quantizers.TernarizationWeightsQuantizer(
    )

  def get_weights_and_quantizers(self, layer):
    return [(getattr(layer, weight_attr), self.weight_quantizer)
            for weight_attr in self.weight_attrs]

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    if len(self.weight_attrs) != len(quantize_weights):
      raise ValueError('`set_quantize_weights` called on layer {} with {} '
                       'weight parameters, but layer expects {} values.'.format(
                           layer.name, len(quantize_weights),
                           len(self.weight_attrs)))

    for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
      current_weight = getattr(layer, weight_attr)
      if current_weight.shape != weight.shape:
        raise ValueError('Existing layer weight shape {} is incompatible with'
                         'provided weight shape {}'.format(
                             current_weight.shape, weight.shape))

      setattr(layer, weight_attr, weight)

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    return []

  @classmethod
  def from_config(cls, config):
    """Instantiates a `TernarizationQuantizeConfig` from its config.

    Args:
        config: Output of `get_config()`.

    Returns:
        A `TernarizationQuantizeConfig` instance.
    """
    return cls(**config)

  def get_config(self):
    # TODO(pulkitb): Add weight and activation quantizer to config.
    # Currently it's created internally, but ideally the quantizers should be
    # part of the constructor and passed in from the registry.
    return {
        'weight_attrs': self.weight_attrs,
    }

  def __eq__(self, other):
    if not isinstance(other, TernarizationQuantizeConfig):
      return False

    return self.weight_attrs == other.weight_attrs

  def __ne__(self, other):
    return not self.__eq__(other)


class TernarizationQuantizeConfigRNN(TernarizationQuantizeConfig, _RNNHelper):
  """QuantizeConfig for RNN layers."""

  def get_weights_and_quantizers(self, layer):
    weights_quantizers = []
    for weight_attrs_cell, rnn_cell in zip(self.weight_attrs,
                                           self._get_rnn_cells(layer)):
      for weight_attr in weight_attrs_cell:
        weights_quantizers.append((getattr(rnn_cell,
                                           weight_attr), self.weight_quantizer))

    return weights_quantizers

  def _flatten(self, list_of_lists):
    flat_list = []
    for sublist in list_of_lists:
      for item in sublist:
        flat_list.append(item)
    return flat_list

  def set_quantize_weights(self, layer, quantize_weights):
    flattened_weight_attrs = self._flatten(self.weight_attrs)
    if len(flattened_weight_attrs) != len(quantize_weights):
      raise ValueError('`set_quantize_weights` called on layer {} with {} '
                       'weight parameters, but layer expects {} values.'.format(
                           layer.name, len(quantize_weights),
                           len(flattened_weight_attrs)))

    i = 0
    for weight_attrs_cell, rnn_cell in zip(self.weight_attrs,
                                           self._get_rnn_cells(layer)):
      for weight_attr in weight_attrs_cell:
        current_weight = getattr(rnn_cell, weight_attr)
        quantize_weight = quantize_weights[i]

        if current_weight.shape != quantize_weight.shape:
          raise ValueError('Existing layer weight shape {} is incompatible with'
                           'provided weight shape {}'.format(
                               current_weight.shape, quantize_weight.shape))

        setattr(rnn_cell, weight_attr, quantize_weight)
        i += 1


class TernarizationConvTransposeQuantizeConfig(TernarizationQuantizeConfig):
  """QuantizeConfig for Conv2DTranspose layers."""

  def __init__(self, weight_attrs):
    super(TernarizationConvTransposeQuantizeConfig, self).__init__(weight_attrs)

    self.weight_quantizer = ternarization_quantizers.TernarizationConvTransposeWeightsQuantizer(
    )


class TernarizationOutputQuantizeConfig(quantize_config.QuantizeConfig):
  """QuantizeConfig which only quantizes the output from a layer."""

  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}


class NoOpQuantizeConfig(quantize_config.QuantizeConfig):
  """QuantizeConfig which does not quantize any part of the layer."""

  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}
