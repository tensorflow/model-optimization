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
"""Abstract base class for clustering algorithm."""

import abc
import six
import tensorflow as tf

from tensorflow_model_optimization.python.core.clustering.keras.cluster_config import GradientAggregation


@six.add_metaclass(abc.ABCMeta)


class ClusteringAlgorithm(object):
  """Class to implement highly efficient vectorised look-ups.

    We do not utilise looping for that purpose, instead we `smartly` reshape and
    tile arrays. The trade-off is that we are potentially using way more memory
    than we would have if looping is used.

    Each class that inherits from this class is supposed to implement a
    particular lookup function for a certain shape.

    For example, look-ups for 2D table will be different in the case of 3D.
  """

  def __init__(
      self,
      clusters_centroids,
      cluster_gradient_aggregation=GradientAggregation.SUM,
      data_format=None
  ):
    """Generating clustered tensors.

    For generating clustered tensors we will need two things: cluster
    centroids and the final shape tensor must have.

    Args:
      clusters_centroids: An array of shape (N,) that contains initial values of
        clusters centroids.
      cluster_gradient_aggregation: An enum that specify the aggregation method
        of the cluster gradient.
      data_format: To be used in cluster_per_channel to ensure the weight
        kernel is permuted properly when updating the weights and calculating
        gradients
    """
    if not isinstance(clusters_centroids, tf.Variable):
      raise ValueError("clusters_centroids should be a tf.Variable.")

    self.cluster_centroids = clusters_centroids
    self.cluster_gradient_aggregation = cluster_gradient_aggregation
    self.data_format = data_format

  @tf.custom_gradient
  def average_centroids_gradient_by_cluster_size(self, cluster_centroids,
                                                 cluster_sizes):
    """Average the gradient based on the number of weights."""

    def grad(d_cluster_centroids):
      # Average the gradient based on the number of weights belonging to each
      # cluster
      d_cluster_centroids = tf.math.divide_no_nan(d_cluster_centroids,
                                                  cluster_sizes)
      return d_cluster_centroids, None

    return cluster_centroids, grad

  @tf.custom_gradient
  def add_gradient_to_original_weight(self, clustered_weight, original_weight):
    """Overrides gradients in the backprop stage.

    This function overrides gradients in the backprop stage: the Jacobian
    matrix of multiplication is replaced with the identity matrix, which
    effectively changes multiplication into add in the backprop. Since
    the gradient of tf.sign is 0, overwriting it with identity follows
    the design of straight-through-estimator, which accepts all upstream
    gradients and uses them to update original non-clustered weights of
    the layer. Here, we assume the gradient updates on individual elements
    inside a cluster will be different so that there is no point in mapping
    the gradient updates back to original non-clustered weights using the LUT.

    Args:
      clustered_weight: clustered weights
      original_weight: original weights

    Returns:
      result and custom gradient, as expected by @tf.custom_gradient
    """
    override_weights = tf.sign(original_weight + 1e+6)
    override_clustered_weight = clustered_weight * override_weights

    def grad(d_override_clustered_weight):
      return d_override_clustered_weight, d_override_clustered_weight

    return override_clustered_weight, grad

  def get_pulling_indices(self, weight, centroids=None):
    """Returns indices of closest cluster centroids.

    Takes a weight(can be 1D, 2D or ND) and creates tf.int32 array of the
    same shape that will hold indices of cluster centroids clustered arrays
    elements will be pulled from.

    In the current setup pulling indices are meant to be created once and
    used everywhere.

    Args:
      weight: ND array of weights. For each weight in this array the closest
        cluster centroids is found.
      centroids: Optional list of cluster centroids.

    Returns:
      ND array of the same shape as `weight` parameter of the type
      tf.int32. The returned array contain weight lookup indices
    """

    cluster_centroids = centroids if centroids is not None else (
        self.cluster_centroids)

    # We find the nearest cluster centroids and store them so that ops can build
    # their kernels upon it.
    pulling_indices = tf.argmin(
        tf.abs(tf.expand_dims(weight, axis=-1) - cluster_centroids),
        axis=-1)

    return pulling_indices

  def get_clustered_weight(self, pulling_indices, original_weight):
    """Returns clustered weights with custom gradients.

    Take indices (pulling_indices) as input and then form a new array
    by gathering cluster centroids based on the given pulling indices.
    The original gradients will also be modified in two ways:
    - By averaging the gradient of cluster_centroids based on the size of
      each cluster.
    - By adding an estimated gradient onto the non-differentiable
      original weight.
    Args:
      pulling_indices: a tensor of indices used for lookup of the same size as
        original_weight.
      original_weight: the original weights of the wrapped layer.

    Returns:
      array with the same shape as `pulling_indices`. Each array element
      is a member of self.cluster_centroids. The backward pass is modified by
      adding custom gradients.
    """

    if self.cluster_gradient_aggregation == GradientAggregation.SUM:
      cluster_centroids = self.cluster_centroids
    elif self.cluster_gradient_aggregation == GradientAggregation.AVG:
      cluster_centroids = self.cluster_centroids
      # Compute the size of each cluster
      # (number of weights belonging to each cluster)
      cluster_sizes = tf.math.bincount(
          arr=tf.cast(pulling_indices, dtype=tf.int32),
          minlength=tf.size(cluster_centroids),
          dtype=cluster_centroids.dtype,
      )
      # Modify the gradient of cluster_centroids to be averaged by cluster sizes
      cluster_centroids = self.average_centroids_gradient_by_cluster_size(
          cluster_centroids,
          tf.stop_gradient(cluster_sizes),
      )
    else:
      raise ValueError(f"self.cluster_gradient_aggregation="
                       f"{self.cluster_gradient_aggregation} not implemented.")

    # Gather the clustered weights based on cluster centroids and
    # pulling indices.
    clustered_weight = tf.gather(cluster_centroids, pulling_indices)

    # Add an estimated gradient to the original weight
    clustered_weight = self.add_gradient_to_original_weight(
        clustered_weight,
        # Fix the bug with MirroredVariable and tf.custom_gradient:
        # tf.identity will transform a MirroredVariable into a Variable
        tf.identity(original_weight),
    )

    return clustered_weight


class ClusteringAlgorithmPerChannel(ClusteringAlgorithm):
  """Class for Per-channel clustering of Conv2D layers."""

  def get_pulling_indices(self, weight):
    """Returns indices of closest cluster centroids.

    This function is based on the function get_pulling_indices
    of the base class ClusteringAlgorithm. We apply each per
    channel of the convolutional layer.

    Args:
      weight: ND array of weights. For each weight in this array the closest
        cluster centroids is found.

    Returns:
      ND array of the same shape as `weight` parameter of the type
      tf.int32. The returned array contain weight lookup indices.
    """
    channel_indices = []

    num_channels = (weight.shape[1] if self.data_format == "channels_first"
                    else weight.shape[-1])

    for channel in range(num_channels):
      channel_weights = (
          weight[:, channel, :, :]
          if self.data_format == "channels_first" else weight[:, :, :, channel])
      channel_centroids = self.cluster_centroids[channel]
      pulling_indices = super().get_pulling_indices(channel_weights,
                                                    channel_centroids)

      channel_indices.append(pulling_indices)

    pulling_indices = tf.convert_to_tensor(channel_indices)
    pulling_indices = tf.transpose(
        pulling_indices,
        perm=(1, 0, 2, 3) if self.data_format == "channels_first" else
        (1, 2, 3, 0))

    return pulling_indices

  def get_clustered_weight(self, pulling_indices, original_weight):
    """Returns clustered weights with custom gradients.

    Take indices the per-channel pulling_indices as input and retrieve
    the corresponding clustered weights by using the gather operation
    for each of the channels.
    The original gradients will also be modified in two ways:
    - By averaging the gradient of cluster_centroids based on the size of
      each cluster.
    - By adding an estimated gradient onto the non-differentiable
      original weight.
    Args:
      pulling_indices: a tensor of per-channel indices used for lookup of the
        same size as original_weight.
      original_weight: the original weights of the wrapped layer.

    Returns:
      array with the same shape as `pulling_indices`. Each array element
      is a member of self.cluster_centroids. The backward pass is modified by
      adding custom gradients.
    """
    num_channels = (
        original_weight.shape[1]
        if self.data_format == "channels_first" else original_weight.shape[-1])

    # In case of channels_last, we have NHWC.
    # In case of channels_first, we have NCHW.
    # We need to transpose the tensor, so C is the first dimension
    # and then we could loop over channels
    pulling_indices = (
        tf.transpose(
            pulling_indices,
            perm=(1, 0, 2, 3) if self.data_format == "channels_first" else
            (3, 0, 1, 2)))

    if self.cluster_gradient_aggregation == GradientAggregation.SUM:
      cluster_centroids = self.cluster_centroids
    elif self.cluster_gradient_aggregation == GradientAggregation.AVG:
      cluster_sizes = []
      for i in range(num_channels):
        # Compute the size of each cluster for each channel
        # (number of weights belonging to each cluster)
        cluster_sizes.append(tf.math.bincount(
            arr=tf.cast(pulling_indices[i], dtype=tf.int32),
            minlength=tf.size(self.cluster_centroids[i]),
            dtype=self.cluster_centroids.dtype,
            ))

      cluster_sizes = tf.convert_to_tensor(cluster_sizes)

      # Modify the gradient of cluster_centroids to be averaged by cluster sizes
      cluster_centroids = self.average_centroids_gradient_by_cluster_size(
          self.cluster_centroids,
          tf.stop_gradient(cluster_sizes),
      )
    else:
      raise ValueError(f"self.cluster_gradient_aggregation="
                       f"{self.cluster_gradient_aggregation} not implemented.")

    clustered_weights = []

    for i in range(num_channels):
      clustered_weights.append(
          tf.gather(cluster_centroids[i], pulling_indices[i]))

    clustered_weight = tf.convert_to_tensor(clustered_weights)

    # Permute weights to ensure the channels are first or last, as expected
    # based on the data_format attribute
    clustered_weight = (
        tf.transpose(clustered_weights, perm=[1, 0, 2, 3])
        if self.data_format == "channels_first" else tf.transpose(
            clustered_weights, perm=[1, 2, 3, 0]))

    # Add an estimated gradient to the original weight
    clustered_weight = self.add_gradient_to_original_weight(
        clustered_weight,
        tf.identity(original_weight),
    )

    return clustered_weight
