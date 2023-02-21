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

import logging
import warnings

import tensorflow as tf

from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras import clustering_registry
from tensorflow_model_optimization.python.core.quantization.keras import quant_ops
from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantizers

layers = tf.keras.layers
K = tf.keras.backend

CLUSTER_CENTROIDS = 'cluster_centroids_tf'
PULLING_INDICES = 'pulling_indices_tf'
ORIGINAL_WEIGHTS = 'ori_weights_vars_tf'
WEIGHT_NAME = 'weight_name'
CLUSTERING_IMPL = 'clst_impl'
CENTROIDS_MASK = 'centroids_mask'
SPARSITY_MASK = 'sparsity_mask'


def get_unique(t):
  """Get unique values and lookup index from N-D tensor.

  Args:
    t: tensor
  Returns:
    centroids (unique values), lookup index (same shape as input tensor)
  Example:
    t:
    ([[1.0, 2.0],
      [2.0, 3.0],
      [3.0, 3.0],
      [1.0, 2.0]]
    )
    centroids(unique values):
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


def get_centroids(layer, weight, data_format):
  """Gets centroid infos from the weights of a layer.

  Args:
    layer: The Keras layer from which the weight belong.
    weight: The weight tensor to get the centroids info from.
    data_format: string to indicate format: "channels_first" or "channels_last".
  Returns:
    A 4-tuple of centroids (unique values), number of centroids, lookup index,
    whether to cluster per channel (boolean).
  """
  cluster_per_channel = (
      layer.layer and isinstance(layer.layer, tf.keras.layers.Conv2D))

  if not cluster_per_channel:
    centroids, index = get_unique(weight)
    return centroids, tf.size(centroids), index, False

  # In case of cluster_per_channel we need to extract
  # unique values (centroids) for each channel.
  num_channels = weight.shape[1 if data_format == 'channels_first' else -1]
  channel_centroids = []
  channel_indices = []
  num_centroids = []

  for channel in range(num_channels):
    channel_weights = weight[:, :, :, channel]
    centroids, indices = get_unique(channel_weights)

    channel_centroids.append(centroids)
    channel_indices.append(indices)
    num_centroids.append(tf.size(centroids))

  max_centroid = max(num_centroids)
  max_diff = max_centroid - min(num_centroids)

  if max_diff > 1:
    centroids, index = get_unique(weight)
    return centroids, tf.size(centroids), index, False

  for i, centroid in enumerate(channel_centroids):
    if num_centroids[i] != max_centroid:
      one_padding = tf.ones([max_centroid - num_centroids[i]])
      channel_centroids[i] = tf.concat([centroid, one_padding], 0)

  centroids = tf.convert_to_tensor(channel_centroids)
  lookup = tf.convert_to_tensor(channel_indices)

  lookup = tf.transpose(
      lookup,
      perm=(1, 0, 2, 3) if data_format == 'channels_first' else (1, 2, 3, 0))

  return centroids, max_centroid, lookup, True


class _ClusterPreserveInfo(object):
  """ClusterPreserveInfo."""

  def __init__(self, weight_attrs, quantize_config_attrs):
    """ClusterPreserveInfo.

    Args:
      weight_attrs: list of cluster preservable weight attributes of layer.
      quantize_config_attrs: list of quantization configuration class name.
    """
    self.weight_attrs = weight_attrs
    self.quantize_config_attrs = quantize_config_attrs


class ClusterPreserveQuantizeRegistry(object):
  """ClusterPreserveQuantizeRegistry is for built-in keras layers."""
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

  _DISABLE_CLUSTER_PRESERVE = frozenset({
      layers.DepthwiseConv2D,
  })

  def __init__(self, preserve_sparsity):
    self._config_quantizer_map = {
        'Default8BitQuantizeConfig':
        ClusterPreserveDefault8BitWeightsQuantizer(preserve_sparsity),
        'Default8BitConvQuantizeConfig':
        ClusterPreserveDefault8BitConvWeightsQuantizer(preserve_sparsity),
    }

  @classmethod
  def _no_trainable_weights(cls, layer):
    """Returns whether this layer has trainable weights.

    Args:
      layer: The layer to check for trainable weights.
    Returns:
      True/False whether the layer has trainable weights.
    """
    return not layer.trainable_weights

  @classmethod
  def _disable_cluster_preserve(cls, layer):
    """Returns whether to disable this layer for preserving clusters.

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

  def apply_cluster_preserve_quantize_config(self, layer, quantize_config):
    """Applies cluster-preserve weight quantizer.

    Args:
      layer: The layer to check for support.
      quantize_config: quantization config for supporting cluster preservation
      on clustered weights
    Returns:
      The quantize_config with addon cluster preserve weight_quantizer.
    """
    if not self.supports(layer):
      raise ValueError('Layer ' + str(layer.__class__) + ' is not supported.')

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
      raise ValueError('Configuration ' +
                       str(quantize_config.__class__.__name__) +
                       ' is not supported for Layer ' + str(layer.__class__) +
                       '.')

    return quantize_config


class Default8bitClusterPreserveQuantizeRegistry(
    ClusterPreserveQuantizeRegistry):
  """Default 8 bit ClusterPreserveQuantizeRegistry."""

  def get_quantize_config(self, layer):
    """Returns the quantization config with weight_quantizer for a given layer.

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

  def __init__(
      self, num_bits, per_axis, symmetric, narrow_range, preserve_sparsity):
    """ClusterPreserveDefaultWeightsQuantizer.

    Args:
      num_bits: Number of bits for quantization
      per_axis: Whether to apply per_axis quantization. The last dimension is
        used as the axis.
      symmetric: If true, use symmetric quantization limits instead of training
        the minimum and maximum of each quantization range separately.
      narrow_range: In case of 8 bits, narrow_range nudges the quantized range
        to be [-127, 127] instead of [-128, 127]. This ensures symmetric
        range has 0 as the centre.
      preserve_sparsity: Whether to apply prune-cluster-preserving quantization
        aware training.
    """
    super(ClusterPreserveDefaultWeightsQuantizer, self).__init__(
        num_bits=num_bits,
        per_axis=per_axis,
        symmetric=symmetric,
        narrow_range=narrow_range,
    )
    self.preserve_sparsity = preserve_sparsity

  def _build_clusters(self, name, layer):
    """Extracts the cluster centroids and cluster indices.

    Extracts cluster centroids and cluster indices from the pretrained
    clustered model when the input layer is clustered.

    Args:
      name: Name of weights in layer.
      layer: Quantization wrapped keras layer.
    Returns:
      A dictionary of the initial values of the
      cluster centroids, cluster indices, original weights,
      the pretrained flag for marking the first training
      epoch, and weight name.
    """
    result = {}
    weights = getattr(layer.layer, name)
    if self.preserve_sparsity and not tf.reduce_any(weights == 0):
      self.preserve_sparsity = False
      logging.warning(
          'Input layer does not contain zero weights, so apply CQAT instead.')
    centroids_mask = None

    # Detects whether layer is convolutional and is clustered per channel
    data_format = getattr(layer.layer, 'data_format', None)
    centroids, num_centroids, lookup, cluster_per_channel = get_centroids(
        layer, weights, data_format)

    if self.preserve_sparsity:
      sparsity_mask = tf.math.divide_no_nan(weights, weights)
      zero_idx = tf.argmin(tf.abs(centroids), axis=-1)
      centroids_mask = 1.0 - tf.one_hot(zero_idx, num_centroids)
      result = {SPARSITY_MASK: sparsity_mask}

    # Prepare clustering variables for the Keras graph when clusters
    # exist, assuming we do not use number_of_clusters larger than 1024
    if num_centroids > 1024:
      warnings.warn(f'No clustering performed on layer {layer.name}.\n'
                    f'Too many centroids to cluster.')
      return result
    # If not enough clusters, we do not preserve clustering
    elif num_centroids <= 1:
      warnings.warn(f'No clustering performed on layer {layer.name}.\n'
                    f'Perhaps too many clusters requested for this layer?')
      return result
    else:
      clst_centroids_tf = layer.add_weight(
          CLUSTER_CENTROIDS,
          shape=centroids.shape,
          initializer=tf.keras.initializers.Constant(
              value=K.batch_get_value([centroids])[0]),
          dtype=centroids.dtype,
          trainable=True)

      ori_weights_tf = layer.add_weight(
          ORIGINAL_WEIGHTS,
          shape=weights.shape,
          initializer=tf.keras.initializers.Constant(
              value=K.batch_get_value([weights])[0]),
          dtype=weights.dtype,
          trainable=True)

      # Get clustering implementation according to layer type
      clustering_impl_cls = clustering_registry.ClusteringLookupRegistry(
      ).get_clustering_impl(
          layer.layer, name, cluster_per_channel=cluster_per_channel)
      clustering_impl = clustering_impl_cls(
          clst_centroids_tf, cluster_config.GradientAggregation.SUM,
          data_format)

      pulling_indices = tf.dtypes.cast(
          clustering_impl.get_pulling_indices(ori_weights_tf),
          lookup.dtype
      )

      pulling_indices_tf = layer.add_weight(
          PULLING_INDICES,
          shape=lookup.shape,
          initializer=tf.keras.initializers.Constant(
              value=K.batch_get_value([pulling_indices])[0]),
          dtype=lookup.dtype,
          trainable=False)

      result_clst = {
          CLUSTER_CENTROIDS: clst_centroids_tf,
          PULLING_INDICES: pulling_indices_tf,
          ORIGINAL_WEIGHTS: ori_weights_tf,
          WEIGHT_NAME: name,
          CLUSTERING_IMPL: clustering_impl,
          CENTROIDS_MASK: centroids_mask,
      }
      result.update(result_clst)
      return result

  def build(self, tensor_shape, name, layer):
    """Build (P)CQAT wrapper.

    When preserve_sparsity is true and the input is clustered.

    Args:
      tensor_shape: Shape of weights which needs to be quantized.
      name: Name of weights in layer.
      layer: Quantization wrapped keras layer.
    Returns:
      Dictionary of centroids, indices and
      quantization params, the dictionary will be passed
      to __call__ function.
    """
    # To get all the initial values from pretrained clustered model
    result = self._build_clusters(name, layer)
    # Result can have clustering nodes, then this is CQAT
    # Result can have both clustering nodes and sparsity mask, then
    # this will be PCQAT
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
    Returns:
      quantized tensor.
    """
    if training:
      if CLUSTER_CENTROIDS in weights:
        if self.preserve_sparsity:
          weights[ORIGINAL_WEIGHTS].assign(
              tf.multiply(weights[ORIGINAL_WEIGHTS],
                          weights[SPARSITY_MASK]))
          weights[CLUSTERING_IMPL].cluster_centroids.assign(
              weights[CLUSTERING_IMPL].
              cluster_centroids * weights[CENTROIDS_MASK]
          )
          weights[CLUSTER_CENTROIDS].assign(
              weights[CLUSTERING_IMPL].cluster_centroids
          )
        # Insert clustering variables
        weights[PULLING_INDICES].assign(tf.dtypes.cast(
            weights[CLUSTERING_IMPL].get_pulling_indices(
                weights[ORIGINAL_WEIGHTS]),
            weights[PULLING_INDICES].dtype
        ))

        output = weights[CLUSTERING_IMPL].get_clustered_weight(
            weights[PULLING_INDICES], weights[ORIGINAL_WEIGHTS])
        inputs.assign(output)
      else:
        if self.preserve_sparsity:
          inputs = tf.multiply(inputs, weights[SPARSITY_MASK])
        output = inputs
    else:
      output = inputs

    return quant_ops.LastValueQuantize(
        output,
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
  """ClusterPreserveWeightsQuantizer for default 8bit weights."""

  def __init__(self, preserve_sparsity):
    super(ClusterPreserveDefault8BitWeightsQuantizer,
          self).__init__(num_bits=8,
                         per_axis=False,
                         symmetric=True,
                         narrow_range=True,
                         preserve_sparsity=preserve_sparsity)
    self.preserve_sparsity = preserve_sparsity


class ClusterPreserveDefault8BitConvWeightsQuantizer(
    ClusterPreserveDefaultWeightsQuantizer,
    default_8bit_quantizers.Default8BitConvWeightsQuantizer):
  """ClusterPreserveWeightsQuantizer for default 8bit Conv2D weights."""

  def __init__(self, preserve_sparsity):  # pylint: disable=super-init-not-called
    default_8bit_quantizers.Default8BitConvWeightsQuantizer.__init__(self)
    self.preserve_sparsity = preserve_sparsity

  def build(self, tensor_shape, name, layer):
    result = ClusterPreserveDefaultWeightsQuantizer._build_clusters(
        self, name, layer)
    result.update(
        default_8bit_quantizers.Default8BitConvWeightsQuantizer.build(
            self, tensor_shape, name, layer))
    return result
