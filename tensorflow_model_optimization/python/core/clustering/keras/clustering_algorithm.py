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


@six.add_metaclass(abc.ABCMeta)
class AbstractClusteringAlgorithm(object):
  """
  The reason to have an abstract class here is to be able to implement highly
  efficient vectorised look-ups.

  We do not utilise looping for that purpose, instead we `smartly` reshape and
  tile arrays. The trade-off is that we are potentially using way more memory
  than we would have if looping is used.

  Each class that inherits from this class is supposed to implement a particular
  lookup function for a certain shape.

  For example, look-ups for 2D table will be different in the case of 3D.
  """

  def __init__(self, clusters_centroids):
    """
    For generating clustered tensors we will need two things: cluster centroids
    and the final shape tensor must have.
    :param clusters_centroids: An array of shape (N,) that contains initial
      values of clusters centroids.
    """
    if not isinstance(clusters_centroids, tf.Variable):
      raise ValueError("clusters_centroids should be a tf.Variable.")

    self.cluster_centroids = clusters_centroids

  @abc.abstractmethod
  def get_pulling_indices(self, weight):
    """
    Takes a weight(can be 1D, 2D or ND) and creates tf.int32 array of the same
    shape that will hold indices of cluster centroids clustered arrays elements
    will be pulled from.

    In the current setup pulling indices are meant to be created once and used
    everywhere
    :param weight: ND array of weights. For each weight in this array the
      closest cluster centroids is found.
    :return: ND array of the same shape as `weight` parameter of the type
      tf.int32. The returned array contain weight lookup indices
    """
    pass

  def get_clustered_weight(self, pulling_indices, original_weight=None):
    """
    Takes indices (pulling_indices) and original weights (weight) as inputs
    and then forms a new array according to the given indices. The original
    weights (weight) here are added to the graph since we want the backprop
    to update their values via the new implementation using tf.custom_gradient
    :param pulling_indices: an array of indices used for lookup.
    :param original_weight: the original weights of the wrapped layer.
      If None, no custom gradient will be added.
    :return: array with the same shape as `pulling_indices`. Each array element
      is a member of self.cluster_centroids
    """
    clustered_weight = tf.gather(self.cluster_centroids, pulling_indices)

    @tf.custom_gradient
    def add_gradient_to_original_weight(clustered_weight, original_weight):
      """
      This function overrides gradients in the backprop stage: the Jacobian
      matrix of multiplication is replaced with the identity matrix, which
      effectively changes multiplication into add in the backprop. Since
      the gradient of tf.sign is 0, overwriting it with identity follows
      the design of straight-through-estimator, which accepts all upstream
      gradients and uses them to update original non-clustered weights of
      the layer. Here, we assume the gradient updates on individual elements
      inside a cluster will be different so that there is no point in mapping
      the gradient updates back to original non-clustered weights using the LUT.
      """
      override_weights = tf.sign(original_weight + 1e+6)
      override_clustered_weight = clustered_weight * override_weights

      def grad(d_override_clustered_weight):
        return d_override_clustered_weight, d_override_clustered_weight

      return override_clustered_weight, grad

    if original_weight is not None:
      clustered_weight = add_gradient_to_original_weight(clustered_weight, original_weight)

    return clustered_weight
