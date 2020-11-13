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
"""Weight clustering algorithm using tfmot compression api."""
from typing import List

import tensorflow as tf

# TODO(tfmot): Make sure weight clustering APIs can be used in this place or
# move the APIs into the same directory.
from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids
from tensorflow_model_optimization.python.core.clustering.keras import clustering_registry
from tensorflow_model_optimization.python.core.common.keras.compression import algorithm


class WeightClusteringParams(object):
  """Weight clustering parameters."""

  def __init__(self,
               number_of_clusters,
               cluster_centroids_init):
    self.number_of_clusters = number_of_clusters
    self.cluster_centroids_init = cluster_centroids_init


class WeightClustering(algorithm.WeightCompressionAlgorithm):
  """Weight clustering compression module config."""

  def __init__(self, params):
    self.params = params

  def init_training_weights_repr(
      self, pretrained_weight: tf.Tensor) -> List[algorithm.WeightRepr]:
    """Init function from pre-trained model case."""
    centroid_initializer = clustering_centroids.CentroidsInitializerFactory.\
        get_centroid_initializer(
            self.params.cluster_centroids_init
        )(pretrained_weight, self.params.number_of_clusters)

    cluster_centroids = centroid_initializer.get_cluster_centroids()

    if len(pretrained_weight.shape) == 2:
      clustering_impl_cls = clustering_registry.DenseWeightsCA
    elif len(pretrained_weight.shape) == 4:
      clustering_impl_cls = clustering_registry.ConvolutionalWeightsCA
    else:
      raise NotImplementedError('Only for dimension=2 or 4 is supported.')

    clustering_impl = clustering_impl_cls(
        cluster_centroids
    )

    # We find the nearest cluster centroids and store them so that ops can
    # build their weights upon it. These indices are calculated once and
    # stored forever. We use to make look-ups from self.cluster_centroids_tf
    pulling_indices = clustering_impl.get_pulling_indices(pretrained_weight)

    return [
        algorithm.WeightRepr(
            name='cluster_centroids',
            shape=cluster_centroids.shape,
            dtype=cluster_centroids.dtype,
            initializer=tf.keras.initializers.Constant(cluster_centroids)),
        algorithm.WeightRepr(
            name='pulling_indices',
            shape=pulling_indices.shape,
            dtype=pulling_indices.dtype,
            initializer=tf.keras.initializers.Constant(pulling_indices))
    ]

  def decompress(self,
                 cluster_centroids: tf.Tensor,
                 pulling_indices: tf.Tensor) -> tf.Tensor:
    return tf.reshape(
        tf.gather(cluster_centroids,
                  tf.reshape(pulling_indices, shape=(-1,))),
        pulling_indices.shape)

  def training(self,
               cluster_centroids: tf.Tensor,
               pulling_indices: tf.Tensor) -> tf.Tensor:
    return self.decompress(cluster_centroids, pulling_indices)

  def get_compressible_weights(
      self, original_layer: tf.keras.layers.Layer) -> List[str]:
    if isinstance(original_layer, tf.keras.layers.Conv2D) or \
       isinstance(original_layer, tf.keras.layers.Dense):
      return ['kernel']
    return []


def optimize(
    to_optimize: tf.keras.Model,
    params: WeightClusteringParams) -> tf.keras.Model:
  """Model developer API for optimizing a model."""

  def _optimize_layer(layer):
    # Require layer to be built so that the SVD-factorized weights
    # can be initialized from the weights.
    if not layer.built:
      raise ValueError(
          'Applying weight clustering currently '
          'requires passing in a built model')

    return algorithm.create_layer_for_training(
        layer, algorithm=WeightClustering(params))

  return tf.keras.models.clone_model(
      to_optimize, clone_function=_optimize_layer)
