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
"""Experimental clustering API functions for Keras models."""

from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras.cluster import _cluster_weights

CentroidInitialization = cluster_config.CentroidInitialization


def cluster_weights(
    to_cluster,
    number_of_clusters,
    cluster_centroids_init=CentroidInitialization.KMEANS_PLUS_PLUS,
    preserve_sparsity=False,
    cluster_per_channel=False,
    **kwargs):
  """Modify a keras layer or model to be clustered during training (experimental).

  This function wraps a keras model or layer with clustering functionality
  which clusters the layer's weights during training. For examples, using
  this with number_of_clusters equals 8 will ensure that each weight tensor has
  no more than 8 unique values.

  Before passing to the clustering API, a model should already be trained and
  show some acceptable performance on the testing/validation sets.

  The function accepts either a single keras layer
  (subclass of `keras.layers.Layer`), list of keras layers or a keras model
  (instance of `keras.models.Model`) and handles them appropriately.

  If it encounters a layer it does not know how to handle, it will throw an
  error. While clustering an entire model, even a single unknown layer would
  lead to an error.

  Cluster a model:

  ```python
  clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': CentroidInitialization.DENSITY_BASED,
    'preserve_sparsity': False
  }

  clustered_model = cluster_weights(original_model, **clustering_params)
  ```

  Cluster a layer:

  ```python
  clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': CentroidInitialization.DENSITY_BASED,
    'preserve_sparsity': False
  }

  model = keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      cluster_weights(layers.Dense(2, activation='tanh'), **clustering_params)
  ])
  ```

  Cluster a layer with sparsity preservation:

  ```python
  clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': CentroidInitialization.DENSITY_BASED,
    'preserve_sparsity': True
  }

  model = keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      cluster_weights(layers.Dense(2, activation='tanh'), **clustering_params)
  ])
  ```

  Arguments:
      to_cluster: A single keras layer, list of keras layers, or a
        `tf.keras.Model` instance.
      number_of_clusters: the number of cluster centroids to form when
        clustering a layer/model. For example, if number_of_clusters=8 then only
        8 unique values will be used in each weight array.
      cluster_centroids_init: `tfmot.clustering.keras.CentroidInitialization`
        instance that determines how the cluster centroids will be initialized.
      preserve_sparsity: optional boolean value that determines whether or not
        sparsity preservation will be enforced during training.
      cluster_per_channel: optional boolean value that determines whether the
        clustering should be applied separately on the individual channels, as
        opposed to the whole kernel. Only applicable to Conv2D layers and is
        ignored otherwise. The number of clusters in this case would be
        num_clusters*num_channels. This is useful for the collaborative
        optimization pipeline where clustering is followed by quantization,
        since Conv2D is quantized per-channel, so we end up with
        num_clusters*num_channels total clusters at the end. Clustering
        per-channel from the beginning leads to better accuracy.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
        Ignored when to_cluster is not a keras layer.

  Returns:
    Layer or model modified to include clustering related metadata.

  Raises:
    ValueError: if the keras layer is unsupported, or the keras model contains
    an unsupported layer.
  """
  return _cluster_weights(to_cluster, number_of_clusters,
                          cluster_centroids_init,
                          preserve_sparsity,
                          cluster_per_channel, **kwargs)
