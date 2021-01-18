# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry

from tensorflow.keras import backend as K
from tensorflow_model_optimization.python.core.quantization.keras.collaborative_optimizations.cluster_preserve import cluster_utils

from tensorflow_model_optimization.python.core.clustering.keras import clustering_registry

layers = tf.keras.layers


def get_unique(t):
  """Get unique values and lookup index from N-D tensor
  Args:
    t: tensor
  Returns:
    unique value, lookup index (same shape as input tensor)
  Example:
    t:
    ([[1.0, 2.0],
      [2.0, 3.0],
      [3.0, 3.0],
      [1.0, 2.0]]
    )
    uniques:
    ([1.0, 2.0, 3.0])
    output final index:
    ([[0, 1],
      [1, 2],
      [2, 2],
      [0, 1]]
    )
  """
  t_flatten = tf.reshape(t, shape=(-1,))
  uniques, index = tf.unique(t_flatten)
  return uniques, tf.reshape(index, shape=tf.shape(t))


class _ClusterPreserveInfo(object):
  def __init__(self, weight_attrs, quantize_config_attrs):
    """ClusterPreserveInfo.
    Args:
      weight_attrs: list of cluster preservable weight attributes of layer.
      quantize_config_attrs: list of quantization configuration class name.
    """
    self.weight_attrs = weight_attrs
    self.quantize_config_attrs = quantize_config_attrs


class ClusterPreserveQuantizeRegistry(object):
  """ClusterPreserveQuantizeRegistry for built-in keras layers."""
  # The keys represent built-in keras layers; the first values represent the
  # the variables within the layers which hold the kernel weights, second
  # values represent the class name of quantization configuration for layers.
  # This decide the weights of layers with quantization configurations are
  # cluster preservable.
  _LAYERS_CONFIG_MAP = {
      layers.Conv2D:
      _ClusterPreserveInfo(['kernel'], ['Default8BitConvQuantizeConfig']),
      layers.Dense:
      _ClusterPreserveInfo(['kernel'], ['Default8BitQuantizeConfig']),

      # DepthwiseConv2D is supported with 8bit qat, but not with
      # clustering, thus for DepthwiseConv2D CQAT,
      # preserving clustered weights is disabled.
      layers.DepthwiseConv2D:
      _ClusterPreserveInfo(['depthwise_kernel'],
                           ['Default8BitQuantizeConfig']),

      # layers that are supported with clustering, but not yet with qat
      # layers.Conv1D:
      # _ClusterPreserveInfo(['kernel'], []),
      # layers.Conv2DTranspose:
      # _ClusterPreserveInfo(['kernel'], []),
      # layers.Conv3D:
      # _ClusterPreserveInfo(['kernel'], []),
      # layers.Conv3DTranspose:
      # _ClusterPreserveInfo(['kernel'], []),
      # layers.LocallyConnected1D:
      # _ClusterPreserveInfo(['kernel'], ['Default8BitQuantizeConfig']),
      # layers.LocallyConnected2D:
      # _ClusterPreserveInfo(['kernel'], ['Default8BitQuantizeConfig']),

      # SeparableConv need verify from 8bit qat
      # layers.SeparableConv1D:
      # _ClusterPreserveInfo(['pointwise_kernel'],
      #                      ['Default8BitConvQuantizeConfig']),
      # layers.SeparableConv2D:
      # _ClusterPreserveInfo(['pointwise_kernel'],
      #                      ['Default8BitConvQuantizeConfig']),

      # Embedding need verify from 8bit qat
      # layers.Embedding: _ClusterPreserveInfo(['embeddings'], []),
  }

  _DISABLE_CLUSTER_PRESERVE = {
      layers.DepthwiseConv2D,
  }

  def __init__(self):
    self._config_quantizer_map = {
        'Default8BitQuantizeConfig':
        ClusterPreserveDefault8BitWeightsQuantizer(),
        'Default8BitConvQuantizeConfig':
        ClusterPreserveDefault8BitConvWeightsQuantizer(),
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
  def _disable_cluster_preserve(cls, layer):
    """Returns whether disable this layer for preserving clusters.

    Args:
      layer: The layer to check for disabling.

    Returns:
      True/False whether disabling this layer for preserving clusters.
    """
    return layer.__class__ in cls._DISABLE_CLUSTER_PRESERVE

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
  def get_cluster_preservable_weights(cls, layer):
    """Get cluster preservable weights from keras layer

    Args:
      layer: instance of keras layer

    Returns:
      List of cluster preservable weights
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
    # layers without trainable weights don't need quantize_config for cqat
    if cls._no_trainable_weights(layer):
      return []

    return cls._LAYERS_CONFIG_MAP[layer.__class__].quantize_config_attrs

  def apply_cluster_preserve_quantize_config(self, layer, quantize_config):
    """ applies cluster-preserve weight quantizer

    Args:
      layer: The layer to check for support.
      quantize_config: quantization config for supporting cluster preservation
      on clustered weights

    Returns:
      The quantize_config with addon cluster preserve weight_quantizer.
    """
    if not self.supports(layer):
      raise ValueError("Layer " + str(layer.__class__) + " is not supported.")

    # Example: ReLU, Softmax, and AveragePooling2D (without trainable weights)
    # DepthwiseConv2D (cluster_preserve is disabled)
    if self._no_trainable_weights(layer) or self._disable_cluster_preserve(
      layer):
      return quantize_config
    # Example: Conv2D, Dense layers
    if quantize_config.__class__.__name__ in self._LAYERS_CONFIG_MAP[
      layer.__class__].quantize_config_attrs:
      quantize_config.weight_quantizer = self._config_quantizer_map[
          quantize_config.__class__.__name__]
    else:
      raise ValueError(
          "Configuration " + str(quantize_config.__class__.__name__) +
          " is not supported for Layer " + str(layer.__class__) + ".")

    return quantize_config


class Default8bitClusterPreserveQuantizeRegistry(
  ClusterPreserveQuantizeRegistry):
  """Default 8 bit ClusterPreserveQuantizeRegistry."""
  def __init__(self):
    super(Default8bitClusterPreserveQuantizeRegistry, self).__init__()

  def get_quantize_config(self, layer):
    """Returns the quantization config with addon cluster
    preserve weight_quantizer for the given layer.

    Args:
      layer: input layer to return quantize config for.

    Returns:
      Returns the quantization config for cluster preserve weight_quantizer.
    """
    quantize_config = (default_8bit_quantize_registry.
                       Default8BitQuantizeRegistry().
                       get_quantize_config(layer))
    cluster_aware_quantize_config = super(
        Default8bitClusterPreserveQuantizeRegistry,
        self).apply_cluster_preserve_quantize_config(layer, quantize_config)

    return cluster_aware_quantize_config


class ClusterPreserveDefaultWeightsQuantizer(quantizers.LastValueQuantizer):
  """Quantize weights while preserving clusters."""
  def __init__(self, num_bits, per_axis, symmetric, narrow_range):
    """ClusterPreserveDefaultWeightsQuantizer

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
    super(ClusterPreserveDefaultWeightsQuantizer, self).__init__(
        num_bits=num_bits,
        per_axis=per_axis,
        symmetric=symmetric,
        narrow_range=narrow_range,
    )

  def _build_clusters(self, name, layer):
    """ Extract the cluster centroids and cluster indices
        from the pretrained clustered model.
    Args:
      name: Name of weights in layer.
      layer: Quantization wrapped keras layer.
    Returns: A dictionary of the initial values of the
      cluster centroids, cluster indices, original weights,
      the pretrained flag for marking the first training
      epoch, and weight name.
    """
    weights = getattr(layer.layer, name)
    centroids, lookup = get_unique(weights)

    # Prepare trainable variables for the Keras graph
    clst_centroids_tf = layer.add_weight(
        'cluster_centroids_tf',
        shape=centroids.shape,
        initializer=tf.keras.initializers.Constant(
            value=K.batch_get_value([centroids])[0]),
        dtype=centroids.dtype,
        trainable=True)

    ori_weights_tf = layer.add_weight(
        'ori_weights_vars_tf',
        shape=weights.shape,
        initializer=tf.keras.initializers.Constant(
            value=K.batch_get_value([weights])[0]),
        dtype=weights.dtype,
        trainable=True)

    # Get clustering implementation according to layer type
    clustering_impl_cls = clustering_registry.ClusteringLookupRegistry().\
        get_clustering_impl(layer.layer, name)
    clustering_impl = clustering_impl_cls(clst_centroids_tf)

    pulling_indices = tf.dtypes.cast(
        clustering_impl.get_pulling_indices(ori_weights_tf),
        lookup.dtype
    )

    pulling_indices_tf = layer.add_weight(
        'pulling_indices_tf',
        shape=lookup.shape,
        initializer=tf.keras.initializers.Constant(
            value=K.batch_get_value([pulling_indices])[0]),
        dtype=lookup.dtype,
        trainable=False)

    for v in layer.weights:
      if 'kernel' in v.name:
        kernel = v

    result = {
        "cluster_centroids_tf": clst_centroids_tf,
        "pulling_indices_tf": pulling_indices_tf,
        "ori_weights_vars_tf": ori_weights_tf,
        "weight_name": name,
        "clst_impl": clustering_impl,
        "set_kernel_weight": kernel,
    }

    return result

  def build(self, tensor_shape, name, layer):
    """Extract centroids and indices to preserve weights clusters.

    Args:
      tensor_shape: Shape of weights which needs to be quantized.
      name: Name of weights in layer.
      layer: Quantization wrapped keras layer.

    Returns: Dictionary of centroids, indices and
      quantization params, the dictionary will be passed
      to __call__ function.
    """
    # To get all the initial values from pretrained clustered model
    result = self._build_clusters(name, layer)
    result.update(
        super(ClusterPreserveDefaultWeightsQuantizer,
              self).build(tensor_shape, name, layer))

    return result

  def __call__(self, inputs, training, weights, **kwargs):
    """Apply cluster preserved quantization to the input tensor.

    Args:
      inputs: Input tensor (layer's weights) to be quantized.
      training: Whether the graph is currently training.
      weights: Dictionary of weights (params) the quantizer can use to
        quantize the tensor (layer's weights). This contains the weights
        created in the `build` function.
      **kwargs: Additional variables which may be passed to the quantizer.

    Returns: quantized tensor.
    """
    # update associations
    if training is True:
      weights['pulling_indices_tf'].assign(
          tf.dtypes.cast(weights['clst_impl']
                         .get_pulling_indices(weights['ori_weights_vars_tf']),
                         weights['pulling_indices_tf'].dtype)
      )

      clustered_inputs = weights['clst_impl'].get_clustered_weight_forward(
          weights['pulling_indices_tf'], weights['ori_weights_vars_tf']
      )
      weights['set_kernel_weight'].assign(clustered_inputs)
    else:
      clustered_inputs = inputs

    return quant_ops.LastValueQuantize(
        clustered_inputs,
        weights['min_var'],
        weights['max_var'],
        is_training=training,
        num_bits=self.num_bits,
        per_channel=self.per_axis,
        symmetric=self.symmetric,
        narrow_range=self.narrow_range
    )


class ClusterPreserveDefault8BitWeightsQuantizer(
  ClusterPreserveDefaultWeightsQuantizer):
  """ClusterPreserveWeightsQuantizer for default 8bit weights"""
  def __init__(self):
    super(ClusterPreserveDefault8BitWeightsQuantizer,
          self).__init__(num_bits=8,
                         per_axis=False,
                         symmetric=True,
                         narrow_range=True)


class ClusterPreserveDefault8BitConvWeightsQuantizer(
  ClusterPreserveDefaultWeightsQuantizer,
  default_8bit_quantizers.Default8BitConvWeightsQuantizer):
  """ClusterPreserveWeightsQuantizer for default 8bit Conv2D weights"""
  def __init__(self):
    default_8bit_quantizers.Default8BitConvWeightsQuantizer.__init__(self)

  def build(self, tensor_shape, name, layer):
    result = ClusterPreserveDefaultWeightsQuantizer._build_clusters(
        self, name, layer)
    result.update(
        default_8bit_quantizers.Default8BitConvWeightsQuantizer.build(
            self, tensor_shape, name, layer))
    return result
