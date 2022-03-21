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
"""Clusters centroids initialization API for Keras clustering API."""

import abc
import six
import tensorflow as tf
from tensorflow.python.ops import clustering_ops
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config

k = tf.keras.backend
CentroidInitialization = cluster_config.CentroidInitialization


@six.add_metaclass(abc.ABCMeta)
class AbstractCentroidsInitialisation:
  """Abstract base class for implementing different cluster centroid initialisation algorithms.

  Must be initialised with a reference to the
  weights and implement the single method below.

  Optionally, zero-centroid initialization (used for sparsity-aware clustering)
  can be enforced by setting the preserve_sparsity option in the clustering
  parameters.
  The procedure is the following:
  1. First, one centroid is set to zero explicitly
  2. The zero-point centroid divides the weights into two intervals: positive
  and negative.
  3. The remaining centroids are proportionally allocated to the two intervals
  4. For each interval (positive and negative), the standard initialization is
  used.
  """

  def __init__(self, weights, number_of_clusters,
               cluster_per_channel=False, data_format='channels_last',
               preserve_sparsity=False):

    # Input checks
    if (data_format != 'channels_first' and data_format != 'channels_last'):
      raise ValueError(
          'The given parameter data_format is not correct: {input}'.format(
              input = data_format))

    self.weights = weights
    self.number_of_clusters = number_of_clusters
    self.cluster_per_channel = cluster_per_channel
    self.data_format = data_format
    self.preserve_sparsity = preserve_sparsity

    if cluster_per_channel:
      self.num_channels = (
          weights.shape[1]
          if self.data_format == 'channels_first' else weights.shape[-1])

  @abc.abstractmethod
  def _calculate_centroids_for_interval(self, weight_interval,
                                        number_of_clusters_for_interval):
    pass

  def _regular_clustering(self):
    # Regular clustering calculates the centroids using all the weights
    cluster_centroids = self._calculate_centroids_for_interval(
        self.weights, self.number_of_clusters)

    return cluster_centroids

  def _per_channel_clustering(self):
    """Implements per channel clustering."""

    channel_centroids = []
    for channel in range(self.num_channels):
      channel_weights = (
          self.weights[:, channel, :, :] if self.data_format == 'channels_first'
          else self.weights[:, :, :, channel])
      channel_centroids.append(
          self._calculate_centroids_for_interval(channel_weights,
                                                 self.number_of_clusters))

    cluster_centroids = tf.convert_to_tensor(channel_centroids)

    return cluster_centroids

  def _zero_centroid_initialization(self, weights_to_cluster=None):
    """The zero-centroid sparsity preservation technique works as follows.

    1. First, one centroid is set to zero explicitly
    2. The zero-point centroid divides the weights into two intervals:
       positive and negative
    3. The remaining centroids are proportionally allocated to the two
       intervals
    4. For each interval (positive and negative), the standard initialization
       is used
    This method is also referred to as sparsity-aware centroid initialization.

    Args:
      weights_to_cluster: Optional list of weights.

    Returns:
      centroids.
    """
    # In the case of per-channel clustering, set weights
    # to be clustered to channel weights.
    weights = (
        weights_to_cluster if weights_to_cluster is not None else self.weights)

    # Zero-point centroid
    zero_centroid = tf.zeros(shape=(1,))

    # Get the negative weights
    negative_weights = tf.boolean_mask(weights,
                                       tf.math.less(weights, 0))
    negative_weights_count = tf.size(negative_weights)

    # Get the positive weights
    positive_weights = tf.boolean_mask(weights,
                                       tf.math.greater(weights, 0))
    positive_weights_count = tf.size(positive_weights)

    # Get the number of non-zero weights
    non_zero_weights_count = negative_weights_count + positive_weights_count

    if tf.math.equal(non_zero_weights_count, 0):
      # No non-zero weights available, simply return the zero-centroid
      return zero_centroid

    # Reduce the number of clusters by one to allow room for the zero-point
    # centroid.
    number_of_non_zero_clusters = self.number_of_clusters - 1

    # Split the non-zero clusters proportionally among negative and positive
    # weights.
    negative_weights_ratio = negative_weights_count / non_zero_weights_count
    number_of_negative_clusters = tf.cast(
        tf.math.round(number_of_non_zero_clusters * negative_weights_ratio),
        dtype=tf.int64)
    number_of_positive_clusters = (
        number_of_non_zero_clusters - number_of_negative_clusters)

    # Calculate the negative centroids
    negative_cluster_centroids = self._calculate_centroids_for_interval(
        negative_weights, number_of_negative_clusters)

    # Calculate the positive centroids
    positive_cluster_centroids = self._calculate_centroids_for_interval(
        positive_weights, number_of_positive_clusters)

    # Put all the centroids together: negative, zero, positive
    centroids = tf.concat(
        [negative_cluster_centroids, zero_centroid, positive_cluster_centroids],
        axis=0)

    return centroids

  def _per_channel_zero_centroid_initialization(self):
    """Per-channel sparsity-preserving centroid initialization.

    The per-channel sparsity-preserving centroid initialization
    works as described in the above method, but applied to each
    channel separately.

    Returns:
      List of cluster centroids.
    """

    channel_centroids = []
    for channel in range(self.num_channels):
      channel_weights = (
          self.weights[:, channel, :, :] if self.data_format == 'channels_first'
          else self.weights[:, :, :, channel])

      zero_centroids = self._zero_centroid_initialization(channel_weights)

      # Put all the centroids together: negative, zero, positive
      channel_centroids.append(zero_centroids)

    cluster_centroids = tf.convert_to_tensor(channel_centroids)

    return cluster_centroids

  def get_cluster_centroids(self):
    """Check whether sparsity preservation should be enforced."""
    if self.preserve_sparsity:
      if self.cluster_per_channel:
        return self._per_channel_zero_centroid_initialization()
      else:
        # Apply the zero-centroid sparsity preservation technique
        return self._zero_centroid_initialization()
    elif self.cluster_per_channel:
      return self._per_channel_clustering()
    else:
      # Perform regular clustering
      return self._regular_clustering()


class LinearCentroidsInitialisation(AbstractCentroidsInitialisation):
  """Spaces cluster centroids evenly in the interval [min(weights), max(weights)]."""

  def _calculate_centroids_for_interval(self, weight_interval,
                                        number_of_clusters_for_interval):
    if tf.math.less_equal(number_of_clusters_for_interval, 0):
      # Return an empty array of centroids
      return tf.constant([])

    weight_min = tf.reduce_min(weight_interval)
    weight_max = tf.reduce_max(weight_interval)
    cluster_centroids = tf.linspace(weight_min, weight_max,
                                    number_of_clusters_for_interval)

    return cluster_centroids


class KmeansPlusPlusCentroidsInitialisation(AbstractCentroidsInitialisation):
  """Cluster centroids based on kmeans++ algorithm."""

  def _calculate_centroids_for_interval(self, weight_interval,
                                        number_of_clusters_for_interval):
    if tf.math.less_equal(number_of_clusters_for_interval, 0):
      # Return an empty array of centroids
      return tf.constant([])

    weights = tf.reshape(weight_interval, [-1, 1])
    cluster_centroids = clustering_ops.kmeans_plus_plus_initialization(
        weights,
        number_of_clusters_for_interval,
        seed=9,
        num_retries_per_sample=-1)

    return tf.reshape(cluster_centroids, [number_of_clusters_for_interval])


class RandomCentroidsInitialisation(AbstractCentroidsInitialisation):
  """Sample centroids randomly and uniformly from the interval [min(weights), max(weights)]."""

  def _calculate_centroids_for_interval(self, weight_interval,
                                        number_of_clusters_for_interval):
    weight_min = tf.reduce_min(weight_interval)
    weight_max = tf.reduce_max(weight_interval)
    cluster_centroids = tf.random.uniform(
        shape=(number_of_clusters_for_interval,),
        minval=weight_min,
        maxval=weight_max,
        dtype=weight_interval.dtype)
    return cluster_centroids


class TFLinearEquationSolver:
  """Solves a linear equation y=ax+b for either y or x.

  The line equation is defined with two points (x1, y1) and (x2,y2)
  """

  def __init__(self, x1, y1, x2, y2):
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2

    # Writing params for y=ax+b
    self.a = (y2 - y1) / tf.maximum(x2 - x1, 0.001)
    self.b = y1 - x1 * ((y2 - y1) / tf.maximum(x2 - x1, 0.001))

  def solve_for_x(self, y):
    """For a given y value, find x at which linear function takes value y.

    Args:
      y: the y value
    Returns:
      the corresponding x value
    """
    return (y - self.b) / self.a

  def solve_for_y(self, x):
    """For a given x value, find y at which linear function takes value x.

    Args:
      x: the x value
    Returns:
      the corresponding y value
    """
    return self.a * x + self.b


class TFCumulativeDistributionFunction:
  """Takes an array and builds cumulative distribution function(CDF)."""

  def __init__(self, weights):
    self.weights = weights

  def get_cdf_value(self, given_weight):
    mask = tf.less_equal(self.weights, given_weight)
    less_than = tf.cast(tf.math.count_nonzero(mask), dtype=tf.float32)
    return less_than / tf.size(self.weights, out_type=tf.float32)


class DensityBasedCentroidsInitialisation(AbstractCentroidsInitialisation):
  r"""Density-based centroids initialisation.

  This initialisation means that we build a cumulative distribution
  function(CDF), then linearly space y-axis of this function then find the
  corresponding x-axis points.

  In order to simplify the implementation, here is
  a plan how it is achieved:
  1. Calculate CDF values at points spaced linearly between weight_min and
  weight_max(e.g. 20 points)
  2. Build an array of values linearly spaced between 0 and 1(probability)
  3. Go through the second array and find segment of CDF that contains this
  y-axis value, \hat{y}
  4. interpolate linearly between those two points, get a line equation y=ax+b
  5. solve equation \hat{y}=ax+b for x. The found x value is a new cluster
  centroid.
  """

  def _get_centroids(self, cdf_x_grid, cdf_values, matching_indices):
    """Interpolates linearly.

    Between every found index using 'i' as the current position and 'i-1' as
    a second point. The value of 'x' is a new cluster centroid.

    Args:
      cdf_x_grid: grid of x values.
      cdf_values: cdf values at grid.
      matching_indices: indices of each

    Returns:
      centroids.
    """

    def get_single_centroid(i):
      i_clipped = tf.minimum(i, tf.size(cdf_values) - 1)
      i_previous = tf.maximum(0, i_clipped - 1)

      x1 = cdf_x_grid[i_clipped]
      x2 = cdf_x_grid[i_previous]
      y1 = cdf_values[i_clipped]
      y2 = cdf_values[i_previous]

      # Check whether interpolation is possible
      if y2 == y1:
        # If there's no delta y it doesn't make sense to try to interpolate
        # the value of x, so just take the lower bound instead
        single_centroid = x1
      else:
        # Interpolate linearly
        s = TFLinearEquationSolver(x1=x1, y1=y1, x2=x2, y2=y2)
        single_centroid = s.solve_for_x(y1)

      return single_centroid

    centroids = k.map_fn(get_single_centroid,
                         matching_indices,
                         dtype=tf.float32)
    return centroids

  def _calculate_centroids_for_interval(self, weight_interval,
                                        number_of_clusters_for_interval):
    if tf.math.less_equal(number_of_clusters_for_interval, 0):
      # Return an empty array of centroids
      return tf.constant([])

    # Get the limits of the weight interval
    weights_min = tf.reduce_min(weight_interval)
    weights_max = tf.reduce_max(weight_interval)

    # Calculate the gap to put at either side of the given interval
    weights_gap = 0.01 if not self.preserve_sparsity else tf.minimum(
        0.01,
        tf.minimum(tf.math.abs(weights_min), tf.math.abs(weights_max)) / 2)

    # Calculating the interpolation nodes for the given weights.
    # A gap is introduced on either side to guarantee that the CDF will have
    # 0 and 1 as the first and last value respectively.
    # The value 30 is a guess, we just need a sufficiently large number here
    # since we are going to interpolate values linearly anyway and the initial
    # guess will drift away. For these reasons we do not really
    # care about the granularity of the lookup
    cdf_x_grid = tf.linspace(weights_min - weights_gap,
                             weights_max + weights_gap, 30)

    # Calculate the centroids within the given interval
    cdf = TFCumulativeDistributionFunction(weights=weight_interval)
    cdf_values = k.map_fn(cdf.get_cdf_value, cdf_x_grid)
    probability_space = tf.linspace(0 + 0.01, 1,
                                    number_of_clusters_for_interval)
    matching_indices = tf.searchsorted(
        sorted_sequence=cdf_values, values=probability_space, side='right')

    centroids = self._get_centroids(cdf_x_grid, cdf_values, matching_indices)
    cluster_centroids = tf.reshape(centroids,
                                   (number_of_clusters_for_interval,))

    return cluster_centroids


class CentroidsInitializerFactory:
  """Factory that creates concrete initializers for factory centroids.

  To implement a custom one, inherit from AbstractCentroidsInitialisation
  and implement all the required methods.

  After this, update CentroidsInitialiserFactory.__initialisers hashtable to
  reflect new methods available.
  """
  _initialisers = {
      CentroidInitialization.LINEAR:
          LinearCentroidsInitialisation,
      CentroidInitialization.RANDOM:
          RandomCentroidsInitialisation,
      CentroidInitialization.DENSITY_BASED:
          DensityBasedCentroidsInitialisation,
      CentroidInitialization.KMEANS_PLUS_PLUS:
          KmeansPlusPlusCentroidsInitialisation,
  }

  @classmethod
  def init_is_supported(cls, init_method):
    return init_method in cls._initialisers

  @classmethod
  def get_centroid_initializer(cls, init_method):
    """Gets centroid initializer.

    Args:
      init_method: a CentroidInitialization value representing the init
      method requested
    Returns:
      A concrete implementation of AbstractCentroidsInitialisation
    Raises:
      ValueError if the requested centroid initialization method is not
      recognised
    """
    if not cls.init_is_supported(init_method):
      raise ValueError(
          'Unknown initialisation method: {init_method}. Allowed values are : '
          '{allowed}'.format(
              init_method=init_method,
              allowed=','.join(cls._initialisers.keys())))

    return cls._initialisers[init_method]
