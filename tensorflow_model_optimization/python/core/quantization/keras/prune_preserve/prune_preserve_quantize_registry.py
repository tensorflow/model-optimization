# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow_model_optimization.python.core.quantization.keras import quant_ops
from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import (
    default_8bit_quantizers, )
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import (
    default_8bit_quantize_registry, )

layers = tf.keras.layers


class _PrunePreserveInfo(object):
  """PrunePreserveInfo."""
  def __init__(self, weight_attrs, quantize_config_attrs):
    """PrunePreserveInfo.

    Args:
      weight_attrs: list of sparsity preservable weight attributes of layer.
      quantize_config_attrs: list of quantization configuration class name.
    """
    self.weight_attrs = weight_attrs
    self.quantize_config_attrs = quantize_config_attrs


class PrunePreserveQuantizeRegistry(object):
  """PrunePreserveQuantizeRegistry responsible for built-in keras layers."""

  # The keys represent built-in keras layers; the first values represent the
  # the variables within the layers which hold the kernel weights, second
  # values represent the class name of quantization configuration for layers.
  # This decide the weights of layers with quantization configurations are
  # sparsity preservable.
  _LAYERS_CONFIG_MAP = {
      layers.Conv2D:
      _PrunePreserveInfo(['kernel'], ['Default8BitConvQuantizeConfig']),
      layers.Dense:
      _PrunePreserveInfo(['kernel'], ['Default8BitQuantizeConfig']),

      # layers that supported with prune, but not yet with qat
      # layers.Conv1D:
      # _PrunePreserveInfo(['kernel'], []),
      # layers.Conv2DTranspose:
      # _PrunePreserveInfo(['kernel'], []),
      # layers.Conv3D:
      # _PrunePreserveInfo(['kernel'], []),
      # layers.Conv3DTranspose:
      # _PrunePreserveInfo(['kernel'], []),
      # layers.LocallyConnected1D:
      # _PrunePreserveInfo(['kernel'], ['Default8BitQuantizeConfig']),
      # layers.LocallyConnected2D:
      # _PrunePreserveInfo(['kernel'], ['Default8BitQuantizeConfig']),

      # DepthwiseCon2D is supported with 8bit qat, but not with prune
      # layers.DepthwiseConv2D:
      # _PrunePreserveInfo(['depthwise_kernel'], ['Default8BitConvQuantizeConfig']),

      # SeparableConv need verify from 8bit qat
      # layers.SeparableConv1D:
      # _PrunePreserveInfo(['pointwise_kernel'], ['Default8BitConvQuantizeConfig']),
      # layers.SeparableConv2D:
      # _PrunePreserveInfo(['pointwise_kernel'], ['Default8BitConvQuantizeConfig']),

      # Embedding need verify from 8bit qat
      # layers.Embedding: _PrunePreserveInfo(['embeddings'], []),
  }

  def __init__(self):

    self._config_quantizer_map = {
        'Default8BitQuantizeConfig':
        PrunePerserveDefault8BitWeightsQuantizer(),
        'Default8BitConvQuantizeConfig':
        PrunePerserveDefault8BitConvWeightsQuantizer(),
    }

  @classmethod
  def _no_trainable_weights(cls, layer):
    """Returns whether this layer has trainable weights.

    Args:
      layer: The layer to check for trainable weights.

    Returns:
      True/False whether the layer has trainable weights.
    """

    return len(layer.trainable_weights) == 0

  @classmethod
  def supports(cls, layer):
    """Returns whether the registry supports this layer type.

    Args:
      layer: The layer to check for support.

    Returns:
      True/False whether the layer type is supported.

    """

    # layers without trainable weights are consider supported,
    # e.g., ReLU, Softmax, and AveragePooling2D.
    if cls._no_trainable_weights(layer):
      return True

    if layer.__class__ in cls._LAYERS_CONFIG_MAP:
      return True

    return False

  @classmethod
  def _weight_names(cls, layer):

    if cls._no_trainable_weights(layer):
      return []

    return cls._LAYERS_CONFIG_MAP[layer.__class__].weight_attrs

  @classmethod
  def get_sparsity_preservable_weights(cls, layer):
    """Get sparsity preservable weights from keras layer

    Args:
      layer: instance of keras layer

    Returns:
      List of sparsity preservable weights
    """
    return [getattr(layer, weight) for weight in cls._weight_names(layer)]

  @classmethod
  def get_suppport_quantize_config_names(cls, layer):
    """Get class name of supported quantize config for layer

    Args:
      layer: instance of keras layer

    Returns:
      List of supported quantize config class name.
    """

    # layers without trainable weights don't need quantize_config for pqat
    if cls._no_trainable_weights(layer):
      return []

    return cls._LAYERS_CONFIG_MAP[layer.__class__].quantize_config_attrs

  def apply_sparsity_preserve_quantize_config(self, layer, quantize_config):
    """ apply weights sparsity preservation

    Args:
      layer: The layer to check for support.
      quantize_config: quantization config to check for support,
        apply sparsity preservation to pruned weights

    Returns:
      Returns quantize_config with addon sparsity preserve weight_quantizer.
    """
    if self.supports(layer):
      if self._no_trainable_weights(layer):
        return quantize_config
      if (quantize_config.__class__.__name__
          in self._LAYERS_CONFIG_MAP[layer.__class__].quantize_config_attrs):
        quantize_config.weight_quantizer = self._config_quantizer_map[
            quantize_config.__class__.__name__]
      else:
        raise ValueError("Configuration " +
                         str(quantize_config.__class__.__name__) +
                         " is not supported for Layer " +
                         str(layer.__class__) + ".")
    else:
      raise ValueError("Layer " + str(layer.__class__) + " is not supported.")

    return quantize_config


class Default8bitPrunePreserveQuantizeRegistry(PrunePreserveQuantizeRegistry):
  """Default 8 bit PrunePreserveQuantizeRegistry."""
  def __init__(self):
    super(Default8bitPrunePreserveQuantizeRegistry, self).__init__()

  def get_quantize_config(self, layer):
    """Returns the quantization config with addon sparsity
    preserve weight_quantizer for the given layer.

    Args:
      layer: input layer to return quantize config for.

    Returns:
      Returns the quantization config with sparsity preserve weight_quantizer.
    """
    quantize_config = default_8bit_quantize_registry.Default8BitQuantizeRegistry(
    ).get_quantize_config(layer)
    prune_aware_quantize_config = super(
        Default8bitPrunePreserveQuantizeRegistry,
        self).apply_sparsity_preserve_quantize_config(layer, quantize_config)

    return prune_aware_quantize_config


class PrunePerserveDefaultWeightsQuantizer(quantizers.LastValueQuantizer):
  """Quantize weights while preserve sparsity."""
  def __init__(self, num_bits, per_axis, symmetric, narrow_range):
    """PrunePerserveDefaultWeightsQuantizer

    Args:
      num_bits: Number of bits for quantization
      per_axis: Whether to apply per_axis quantization. The last dimension is
        used as the axis.
      symmetric: If true, use symmetric quantization limits instead of training
        the minimum and maximum of each quantization range separately.
      narrow_range: In case of 8 bits, narrow_range nudges the quantized range
        to be [-127, 127] instead of [-128, 127]. This ensures symmetric
        range has 0 as the centre.
    """

    super(PrunePerserveDefaultWeightsQuantizer, self).__init__(
        num_bits=num_bits,
        per_axis=per_axis,
        symmetric=symmetric,
        narrow_range=narrow_range,
    )

  def _build_sparsity_mask(self, name, layer):
    weights = getattr(layer.layer, name)
    sparsity_mask = tf.math.divide_no_nan(weights, weights)

    return {'sparsity_mask': sparsity_mask}

  def build(self, tensor_shape, name, layer):
    """Construct mask to preserve weights sparsity.

    Args:
      tensor_shape: Shape of weights which needs to be quantized.
      name: Name of weights in layer.
      layer: quantization wrapped keras layer.

    Returns: Dictionary of constructed sparsity mask and
      quantization params, the dictionary will be passed
      to __call__ function.
    """
    result = self._build_sparsity_mask(name, layer)
    result.update(
        super(PrunePerserveDefaultWeightsQuantizer,
              self).build(tensor_shape, name, layer))
    return result

  def __call__(self, inputs, training, weights, **kwargs):
    """Apply sparsity preserved quantization to the input tensor.

    Args:
      inputs: Input tensor (layer's weights) to be quantized.
      training: Whether the graph is currently training.
      weights: Dictionary of weights (params) the quantizer can use to
        quantize the tensor (layer's weights). This contains the weights
        created in the `build` function.
      **kwargs: Additional variables which may be passed to the quantizer.

    Returns: quantized tensor.
    """

    prune_preserve_inputs = tf.multiply(inputs, weights['sparsity_mask'])

    return quant_ops.LastValueQuantize(
        prune_preserve_inputs,
        weights['min_var'],
        weights['max_var'],
        is_training=training,
        num_bits=self.num_bits,
        per_channel=self.per_axis,
        symmetric=self.symmetric,
        narrow_range=self.narrow_range,
    )


class PrunePerserveDefault8BitWeightsQuantizer(
    PrunePerserveDefaultWeightsQuantizer):
  """PrunePerserveWeightsQuantizer for default 8bit weights"""
  def __init__(self):
    super(PrunePerserveDefault8BitWeightsQuantizer,
          self).__init__(num_bits=8,
                         per_axis=False,
                         symmetric=True,
                         narrow_range=True)


class PrunePerserveDefault8BitConvWeightsQuantizer(
    PrunePerserveDefaultWeightsQuantizer,
    default_8bit_quantizers.Default8BitConvWeightsQuantizer,
):
  """PrunePerserveWeightsQuantizer for default 8bit Conv2D/DepthwiseConv2D weights"""
  def __init__(self):
    default_8bit_quantizers.Default8BitConvWeightsQuantizer.__init__(self)

  def build(self, tensor_shape, name, layer):
    result = PrunePerserveDefaultWeightsQuantizer._build_sparsity_mask(
        self, name, layer)
    result.update(
        default_8bit_quantizers.Default8BitConvWeightsQuantizer.build(
            self, tensor_shape, name, layer))
    return result
