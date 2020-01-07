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

import tensorflow.compat.v1 as tf
from absl.testing import parameterized

keras = tf.keras
errors_impl = tf.errors
layers = keras.layers
test = tf.test

from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids
import tensorflow.keras.backend as K


class ClusteringCentroidsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    self.factory = clustering_centroids.CentroidsInitializerFactory
    pass

  @parameterized.parameters(
    ('linear'),
    ('random'),
    ('density-based'),
  )
  def testExistingInitsAreSupported(self, init_type):
    self.assertTrue(self.factory.init_is_supported(init_type))

  def testNonExistingInitIsNotSupported(self):
    self.assertFalse(self.factory.init_is_supported("DEADBEEF"))

  @parameterized.parameters(
    ('linear', clustering_centroids.LinearCentroidsInitialisation),
    ('random', clustering_centroids.RandomCentroidsInitialisation),
    ('density-based', clustering_centroids.DensityBasedCentroidsInitialisation),
  )
  def testReturnsMethodForExistingInit(self, init_type, method):
    self.assertEqual(self.factory.get_centroid_initializer(init_type), method)

  def testThrowsValueErrorForNonExistingInit(self):
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
    solver = clustering_centroids.TFLinearEquationSolver(float(x1), float(y1), float(x2), float(y2))
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
    solver = clustering_centroids.TFLinearEquationSolver(float(x1), float(y1), float(x2), float(y2))
    for_x = solver.solve_for_x(y)
    self.assertAlmostEqual(K.batch_get_value([for_x])[0], x)

  @parameterized.parameters(
    (0, 0, 1, 1, 5, 5),
    (0, 0, 5, 5, 20, 20),
    (1, 2, 3, 4, 3, 4),
    (7, 12, 17, 22, 3, 8),
  )
  def testLinearSolverSolveForY(self, x1, y1, x2, y2, x, y):
    solver = clustering_centroids.TFLinearEquationSolver(float(x1), float(y1), float(x2), float(y2))
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
    cdf_calc = clustering_centroids.TFCumulativeDistributionFunction(weights)
    self.assertAlmostEqual(probability, K.batch_get_value([cdf_calc.get_cdf_value(point)])[0])

  @parameterized.parameters(
    ([0, 1, 2, 3, 3.1, 3.2, 3.3, 3.4, 3.5], 5, [0.11137931, 2.0534482, 3.145862, 3.3886206, 3.51]),
    ([0, 1, 2, 3, 3.1, 3.2, 3.3, 3.4, 3.5], 3, [0.11137931, 3.145862, 3.51]),
    ([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], 3, [0.3010345, 5.2775865, 9.01]),
  )
  def testClusterCentroids(self, weights, number_of_clusters, centroids):
    dbci = clustering_centroids.DensityBasedCentroidsInitialisation(weights, number_of_clusters)
    calc_centroids = K.batch_get_value([dbci.get_cluster_centroids()])[0]
    self.assertSequenceAlmostEqual(centroids, calc_centroids, places=4)


if __name__ == '__main__':
  test.main()
