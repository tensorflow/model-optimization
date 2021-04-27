# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Clusterable layer API class for Keras models."""

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class ClusterableLayer:
  """Abstract Base Class for making your own keras layer clusterable.

  Your layer could be derived from a keras built-in layer or
  it could be a keras custom layer.

  The function get_clusterable_weights should be provided in both cases.

  The function get_clusterable_algorithm is provided, when weights for
  clustering is added in the keras layer.

  """

  @abc.abstractmethod
  def get_clusterable_weights(self):
    """Returns list of clusterable weight tensors.

    All the weight tensors which the layer wants to be clustered during
    training must be returned by this method.

    Returns: List of weight tensors/kernels in the keras layer which must be
        clustered during training. Each element in the list is a (name, kernel)
        2-tuple that consists of the name of the clusterable kernel and the
        kernel object itself.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  def get_clusterable_algorithm(self, weight_name):  # pylint: disable=unused-argument
    """Returns class with the clustering algorithm for the given weight_name.

    This function needs to be implemented for the customerable layers.
    If the layer is derived from the built-in keras layer, the clustering
    algorithm for the base built-in keras layer is used.

    The returned class should be derived from ClusteringAlgorithm and
    implements the function get_pulling_indices.
    This function is used to provide a special lookup function for the custom
    weights.
    It reshapes and tile centroids the same way as the weights. This allows us
    to find pulling indices efficiently.

    Args:
        weight_name ([string]): The name of the weight variable.
    """
    return None
