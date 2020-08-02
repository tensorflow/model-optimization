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
"""Tests for keras clustering centroids initialisation API."""

import tensorflow as tf
import tensorflow.keras.backend as K

from absl.testing import parameterized

from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids

keras = tf.keras
errors_impl = tf.errors
layers = keras.layers
test = tf.test

CentroidInitialization = cluster_config.CentroidInitialization


class ClusteringCentroidsTest(test.TestCase, parameterized.TestCase):
  """Unit tests for the clustering_centroids module."""

  def setUp(self):
    self.factory = clustering_centroids.CentroidsInitializerFactory

  @parameterized.parameters(
      (CentroidInitialization.LINEAR),
      (CentroidInitialization.RANDOM),
      (CentroidInitialization.DENSITY_BASED),
      (CentroidInitialization.KMEANS_PLUS_PLUS),
  )
  def testExistingInitsAreSupported(self, init_type):
    """
    Verifies that the given centroid initialization methods are supported.
    """
    self.assertTrue(self.factory.init_is_supported(init_type))

  def testNonExistingInitIsNotSupported(self):
    self.assertFalse(self.factory.init_is_supported("DEADBEEF"))

  @parameterized.parameters(
      (
          CentroidInitialization.LINEAR,
          clustering_centroids.LinearCentroidsInitialisation
      ),
      (
          CentroidInitialization.RANDOM,
          clustering_centroids.RandomCentroidsInitialisation
      ),
      (
          CentroidInitialization.DENSITY_BASED,
          clustering_centroids.DensityBasedCentroidsInitialisation
      ),
      (
          CentroidInitialization.KMEANS_PLUS_PLUS,
          clustering_centroids.KmeansPlusPlusCentroidsInitialisation
       ),
  )
  def testReturnsMethodForExistingInit(self, init_type, method):
    """
    Verifies that the centroid initializer factory method returns the expected
    classes for the given initialization methods.
    """
    self.assertEqual(self.factory.get_centroid_initializer(init_type), method)

  def testThrowsValueErrorForNonExistingInit(self):
    """
    Verifies that the centroid initializer factory method raises an exception
    when invoked with an unsupported initialization method.
    """
    with self.assertRaises(ValueError):
      self.factory.get_centroid_initializer("DEADBEEF")

  @parameterized.parameters(
      (0, 0, 1, 1, 1, 0),
      (0, 0, 5, 5, 1, 0),
      (1, 2, 3, 4, 1, 1),
      (7, 12, 17, 22, 1, 5),
      (-5, 4, 7, 10, 1.0 / 2.0, 13.0 / 2.0),
  )
  def testLinearSolverConstruction(self, x1, y1, x2, y2, a, b):
    """
    Verifies that a TFLinearEquationSolver is constructed correctly.
    """
    solver = clustering_centroids.TFLinearEquationSolver(float(x1),
                                                         float(y1),
                                                         float(x2),
                                                         float(y2))
    solver_a = solver.a
    self.assertAlmostEqual(K.batch_get_value([solver_a])[0], a)
    self.assertAlmostEqual(K.batch_get_value([solver.b])[0], b)

  @parameterized.parameters(
      (0, 0, 1, 1, 5, 5),
      (0, 0, 5, 5, 20, 20),
      (1, 2, 3, 4, 3, 4),
      (7, 12, 17, 22, 3, 8),
  )
  def testLinearSolverSolveForX(self, x1, y1, x2, y2, x, y):
    """
    Verifies that TFLinearEquationSolver solves the given equations correctly
    for X.
    """
    solver = clustering_centroids.TFLinearEquationSolver(float(x1),
                                                         float(y1),
                                                         float(x2),
                                                         float(y2))
    for_x = solver.solve_for_x(y)
    self.assertAlmostEqual(K.batch_get_value([for_x])[0], x)

  @parameterized.parameters(
      (0, 0, 1, 1, 5, 5),
      (0, 0, 5, 5, 20, 20),
      (1, 2, 3, 4, 3, 4),
      (7, 12, 17, 22, 3, 8),
  )
  def testLinearSolverSolveForY(self, x1, y1, x2, y2, x, y):
    """
    Verifies that TFLinearEquationSolver solves the given equations correctly
    for Y.
    """
    solver = clustering_centroids.TFLinearEquationSolver(float(x1),
                                                         float(y1),
                                                         float(x2),
                                                         float(y2))
    for_y = solver.solve_for_y(x)
    self.assertAlmostEqual(K.batch_get_value([for_y])[0], y)

  @parameterized.parameters(
      ([1, 2, 6, 7], 4, 0.5),
      ([1, 2, 6, 7], 1, 1. / 4.),
      ([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 1. / 3.),
      ([1, 2, 3, 4, 5, 6, 7, 8, 9], 99, 1.),
      ([1, 2, 3, 4, 5, 6, 7, 8, 9], -20, 0.)
  )
  def testCDFValues(self, weights, point, probability):
    """
    Verifies that TFCumulativeDistributionFunction yields the expected output
    for the inputs provided.
    """
    cdf_calc = clustering_centroids.TFCumulativeDistributionFunction(weights)
    self.assertAlmostEqual(
        probability,
        K.batch_get_value([cdf_calc.get_cdf_value(point)])[0]
    )

  @parameterized.parameters(
      (
          [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
          5,
          [0., 2.5, 5., 7.5, 10.]
      ),
      (
          [0., 1., 2., 3., 3.1, 3.2, 3.3, 3.4, 3.5],
          3,
          [0., 1.75, 3.5]
      ),
      (
          [-3., -2., -1., 0., 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.],
          6,
          [-3., -0.6, 1.8, 4.2, 6.6, 9.]
      )
  )
  def testLinearClusterCentroids(self, weights, number_of_clusters, centroids):
    dbci = clustering_centroids.LinearCentroidsInitialisation(
        weights,
        number_of_clusters
    )
    calc_centroids = K.batch_get_value([dbci.get_cluster_centroids()])[0]
    self.assertSequenceAlmostEqual(centroids, calc_centroids, places=4)

  @parameterized.parameters(
      (
          [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
          5,
          [0., 1., 4., 7., 10.]
      ),
      (
          [0., 1., 2., 3., 3.1, 3.2, 3.3, 3.4, 3.5],
          3,
          [0., 1., 3.5]
      ),
      (
          [-3., -2., -1., 0., 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.],
          6,
          [-3., 0., 1.1, 3.7333333, 6.366666, 9.]
      )
  )
  def testLinearClusterCentroidsWithSparsityPreservation(self, weights, number_of_clusters, centroids):
    dbci = clustering_centroids.LinearCentroidsInitialisation(
        weights,
        number_of_clusters,
        True
    )
    calc_centroids = K.batch_get_value([dbci.get_cluster_centroids()])[0]
    self.assertSequenceAlmostEqual(centroids, calc_centroids, places=4)

  @parameterized.parameters(
      (
          [0, 1, 2, 3, 3.1, 3.2, 3.3, 3.4, 3.5],
          5,
          [0.11137931, 2.0534482, 3.145862, 3.3886206, 3.51]
      ),
      (
          [0, 1, 2, 3, 3.1, 3.2, 3.3, 3.4, 3.5],
          3,
          [0.11137931, 3.145862, 3.51]
      ),
      (
          [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
          3,
          [0.3010345, 5.2775865, 9.01]
      )
  )
  def testDensityBasedClusterCentroids(self, weights, number_of_clusters, centroids):
    dbci = clustering_centroids.DensityBasedCentroidsInitialisation(
        weights,
        number_of_clusters
    )
    calc_centroids = K.batch_get_value([dbci.get_cluster_centroids()])[0]
    self.assertSequenceAlmostEqual(centroids, calc_centroids, places=4)

  @parameterized.parameters(
      (
          [0., -1., -2., -3., -4., -5., -6.],
          4,
          [-5.836897, -2.8941379, -0.98999995, 0.]
      ),
      (
          [0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
          5,
          [0., 1.2665517, 4.032069, 7.0741386, 9.01]
      ),
      (
          [-4., -3., -2., -1., 0., 0., 0., 1., 2., 3., 4., 5., 6., 7.],
          6,
          [-3.9058623, -0.99,  0., 1.1975863, 4.103793, 7.01]
      ),
      (
          [0., 1., 2., 3., -3.1, -3.2, -3.3, -0.005, 3.5],
          3,
          [-3.1887069, 0., 1.0768965]
      ),
      (
          [0., 0., 0., 0.],
          2,
          [0.]
      )
  )
  def testDensityBasedClusterCentroidsWithSparsityPreservation(
      self, weights, number_of_clusters, centroids):
    dbci = clustering_centroids.DensityBasedCentroidsInitialisation(
        weights,
        number_of_clusters,
        True
    )
    calc_centroids = K.batch_get_value([dbci.get_cluster_centroids()])[0]
    self.assertSequenceAlmostEqual(centroids, calc_centroids, places=4)

  @parameterized.parameters(
      (
          [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
          5
      ),
      (
          [0., 1., 2., 3., 3.1, 3.2, 3.3, 3.4, 3.5],
          3
      ),
      (
          [-3., -2., -1., 0., 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.],
          6
      )
  )
  def testRandomClusterCentroidsWithSparsityPreservation(self, weights, number_of_clusters):
    dbci = clustering_centroids.RandomCentroidsInitialisation(
        weights,
        number_of_clusters,
        True
    )
    calc_centroids = K.batch_get_value([dbci.get_cluster_centroids()])[0]
    self.assertContainsSubset([0.], calc_centroids, msg="The centroids must include the zero-point cluster")

  @parameterized.parameters(
    (
            [0, 1, 2, 3, 3.1, 3.2, 3.3, 3.4, 3.5],
            5,
            [3.1, 0., 2., 1., 3.4]
    ),
    (
            [0, 1, 2, 3, 3.1, 3.2, 3.3, 3.4, 3.5],
            3,
            [3.1, 0., 2.]
    ),
    (
            [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
            3,
            [6., 1., 8.]
    )
  )
  def testKmeansPlusPlusClusterCentroids(self, weights, number_of_clusters, centroids):
    kmci = clustering_centroids.KmeansPlusPlusCentroidsInitialisation(
        weights,
        number_of_clusters
    )
    calc_centroids = K.batch_get_value([kmci.get_cluster_centroids()])[0]
    self.assertSequenceAlmostEqual(centroids, calc_centroids, places=4)

  @parameterized.parameters(
    (
            [0, 1, 2, 3, 3.1, 3.2, 3.3, 3.4, 3.5],
            5,
            [0., 3., 1., 2., 3.3]
    ),
    (
            [0, 1, 2, 3, 3.1, 3.2, 3.3, 3.4, 3.5],
            3,
            [0., 3., 1.]
    ),
    (
            [-4., -3., -2., -1., 0., 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.],
            6,
            [-2., -4., 0., 5.5, 2.2, 8.8]
    )
  )
  def testKmeansPlusPlusClusterCentroidsWithSparsityPreservation(self, weights, number_of_clusters, centroids):
    kmci = clustering_centroids.KmeansPlusPlusCentroidsInitialisation(
        weights,
        number_of_clusters,
        True
    )
    calc_centroids = K.batch_get_value([kmci.get_cluster_centroids()])[0]
    self.assertSequenceAlmostEqual(centroids, calc_centroids, places=4)

if __name__ == '__main__':
  test.main()
