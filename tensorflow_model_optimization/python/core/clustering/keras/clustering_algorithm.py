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

  @tf.custom_gradient
  def add_custom_gradients(self, clst_weights, weights):
    """
    This function overrides gradients in the backprop stage: original mul
    becomes add, tf.sign becomes tf.identity. It is to update the original
    weights with the gradients updates directly from the layer wrapped. We
    assume the gradients updates on individual elements inside a cluster
    will be different so that there is no point of mapping the gradient
    updates back to original weight matrix using the LUT.
    """
    override_weights = tf.sign(tf.reshape(weights, shape=(-1,)) + 1e+6)
    z = clst_weights*override_weights
    def grad(dz):
      return dz, dz
    return z, grad

  def get_clustered_weight(self, pulling_indices):
    """
    Takes an array with integer number that represent lookup indices and forms a
    new array according to the given indices.
    :param pulling_indices: an array of indices used for lookup.
    :return: array with the same shape as `pulling_indices`. Each array element
      is a member of self.cluster_centroids
    """
    return tf.reshape(
        tf.gather(self.cluster_centroids,
                  tf.reshape(pulling_indices, shape=(-1,))),
        shape=pulling_indices.shape
    )

  def get_clustered_weight_forward(self, pulling_indices, weight):
    """
    Takes indices (pulling_indices) and original weights (weight) as inputs
    and then forms a new array according to the given indices. The original
    weights (weight) here are added to the graph since we want the backprop
    to update their values via the new implementation using tf.custom_gradient
    :param pulling_indices: an array of indices used for lookup.
    :param weight: the original weights of the wrapped layer.
    :return: array with the same shape as `pulling_indices`. Each array element
      is a member of self.cluster_centroids
    """
    x = tf.reshape(self.get_clustered_weight(pulling_indices), shape=(-1,))
    return tf.reshape(self.add_custom_gradients(
        x, tf.reshape(weight, shape=(-1,))), pulling_indices.shape)
