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
"""Tests for the utilities in sparse training pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import itertools
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.sparsity_tf2 import sparse_utils

dtypes = tf.dtypes
test = tf.test


def sample_noise(shape=(), mu=0, sigma=1.):
  sample = tf.random.normal(shape, mean=mu,  stddev=sigma, dtype=tf.float64)
  return sample

def calculate_stdev(p):
  # Bernouilli distribution
  return np.sqrt(p * (1 - p))

RATIOS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
SHAPES = [(), [1], [5], [2, 2], [4, 4], [8, 4], [3, 17], [3, 6, 9]]
NOSMALL_SHAPES = [[5], [2, 2], [4, 4], [8, 4], [3, 17], [3, 6, 9]]
MIDPOINTS = [2., 1.5, 7.5, 15.5, 25., 80.5]

class SparseUtilsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(SparseUtilsTest, self).setUp()
    self.seeds = range(20)

  @parameterized.parameters(
    (-1.,), (1.5,)
  )
  def testInvalidRatioRaisesError(self, p):
    with self.assertRaises(ValueError):
      bernouilli = sparse_utils.Bernouilli(p)
      permuteones = sparse_utils.PermuteOnes(p)

  @parameterized.parameters(
    (tf.int32,), (tf.float32,), (tf.int64,), (tf.float64,)
  )
  def testMatrixDtype(self, dtype):
    shape = (3, 4)
    seed = 0
    ratio = 0.5
    type_bern_checks = []
    type_perm_checks = []
    _dtype = [dtype]
    matrix = tf.ones(shape)
    sparse_bernouilli = sparse_utils.Bernouilli(ratio)(shape=matrix.shape, dtype=dtype, seed=seed)
    type_bern_checks.append(sparse_bernouilli.dtype)
    self.assertAllEqual(_dtype, type_bern_checks)
    sparse_permuteones = sparse_utils.PermuteOnes(ratio)(shape=matrix.shape, dtype=dtype, seed=seed)
    type_perm_checks.append(sparse_permuteones.dtype)
    self.assertAllEqual(_dtype, type_perm_checks)
    
  @parameterized.parameters(
    itertools.product((0.4,), NOSMALL_SHAPES)
  )
  def testScatteredInitialOnesPermuteOnes(self, ratio, shape):
    # This test checks that the expected sparsity is reflected in the 
    # average number of times each index in the matrix is a 1.
    # note that ratio here is fixed because the shape / density
    # would make a greater difference in this check for "scattered-ness".
    # Shapes cannot be small (i.e. () or (1) ) as there are not enough
    # samples to average over.
    matrix = tf.ones(shape)
    self.assertAllEqual(shape, matrix.shape)

    seed = 0
    sparse_matrices_perm = []
    for _ in range(400):
      matrix_sparse_permute = sparse_utils.PermuteOnes(ratio)(shape, seed=seed)
      sparse_matrices_perm.append(matrix_sparse_permute)
      seed += 1
    average_initialization = tf.math.reduce_mean(sparse_matrices_perm, axis=0)
    expected_initialization = tf.constant(ratio, shape=shape) # broadcasts
    # expect that each entry in the matrix is "on"/1 in expectation, ratio number of times
    self.assertAllClose(expected_initialization, average_initialization, rtol=231e-3)
   
  @parameterized.parameters(
    itertools.product((0.4,), NOSMALL_SHAPES)
  )
  def testScatteredInitialOnesBernouilli(self, ratio, shape):
    matrix = tf.ones(shape)
    self.assertAllEqual(shape, matrix.shape)

    seed = 0
    sparse_matrices_bern = []
    for _ in range(400):
      matrix_sparse_bernouilli = sparse_utils.Bernouilli(ratio)(shape, seed=seed)
      sparse_matrices_bern.append(matrix_sparse_bernouilli)
      seed += 1
    average_initialization = tf.math.reduce_mean(sparse_matrices_bern, axis=0)
    expected_initialization = tf.constant(ratio, shape=shape)
    self.assertAllClose(expected_initialization, average_initialization, rtol=204e-3)

  @parameterized.parameters(
    zip(itertools.product((0.4,), NOSMALL_SHAPES), MIDPOINTS)
  )
  def testMidpointInitialOnesPermuteOnes(self, ratio_shape_pair, midpoint):
    # Given all indices that are masked according to ratio,
    # the average value of all indices should appear at the "center"
    # of the given matrix. Similarly to above test, the ratio
    # of 1's in the mask is not deciding factor.
    ratio, shape = ratio_shape_pair
    matrix = tf.ones(shape)
    self.assertAllEqual(shape, matrix.shape)

    seed = 0
    sparse_matrices_perm = []
    for _ in range(400):
      matrix_sparse_permute = sparse_utils.PermuteOnes(ratio)(shape, seed=seed)
      matrix_midpoint_idx = tf.where(tf.reshape(matrix_sparse_permute, (-1,)) == 1)
      sparse_matrices_perm.append(tf.math.reduce_mean(matrix_midpoint_idx))
      seed += 1
    average_midpoints = tf.math.reduce_mean(sparse_matrices_perm, axis=0)
    expected_midpoints = tf.constant(midpoint) # broadcasts
    # expect that each entry in the matrix is "on"/1 in expectation, ratio number of times
    self.assertAllClose(expected_midpoints, average_midpoints, atol=1.)

  @parameterized.parameters(
    zip(itertools.product((0.4,), NOSMALL_SHAPES), MIDPOINTS)
  )
  def testMidpointInitialOnesBernouilli(self, ratio_shape_pair, midpoint):
    ratio, shape = ratio_shape_pair
    matrix = tf.ones(shape)
    self.assertAllEqual(shape, matrix.shape)

    seed = 0
    sparse_matrices_bern = []
    for _ in range(400):
      matrix_sparse_bernouilli = sparse_utils.Bernouilli(ratio)(shape, seed=seed)
      matrix_midpoint_idx = tf.where(tf.reshape(matrix_sparse_bernouilli, (-1,)) == 1)
      sparse_matrices_bern.append(tf.math.reduce_mean(matrix_midpoint_idx))
      seed += 1
    average_midpoint = tf.math.reduce_mean(sparse_matrices_bern, axis=0)
    expected_midpoint = tf.constant(midpoint)
    self.assertAllClose(expected_midpoint, average_midpoint, atol=1.)
  

  @parameterized.parameters(
    ((30, 4), 0.5, 60), 
    ((1, 2, 1, 4), 0.8, 7), 
    ((30,), 0.1, 3)
  )
  def testBernouilliMatrixFraction(self, shape, ratio, expected_ones):
    # Check that the mean of ones_count is approximately
    # ratio * num_elements_in_shape.
    matrix = tf.ones(shape)
    self.assertAllEqual(shape, matrix.shape)

    seed = 0

    counts = []
    for _ in range(400):
      output = sparse_utils.Bernouilli(ratio)(shape, seed=seed)
      ones_count = tf.math.count_nonzero(output, dtype=tf.int32)
      counts.append(ones_count)
      self.assertAllEqual(output.shape, shape)
      seed += 1
    mean_counts = np.mean(counts, axis=0)
    stdev_counts = np.std(counts, axis=0)
    self.assertAllClose(mean_counts, expected_ones, rtol=8e-2)

  @parameterized.parameters(
    ((30, 4), 0.5, 60), 
    ((1, 2, 1, 4), 0.8, 7), 
    ((30,), 0.1, 3)
  )
  def testDeterministicMatrixFraction(self, shape, ratio, expected_ones):
    matrix = tf.ones(shape)
    self.assertAllEqual(shape, matrix.shape)

    for seed in self.seeds:
      output = sparse_utils.PermuteOnes(ratio)(shape, seed=seed)
      self.assertAllEqual(output.shape, shape)
      ones_count = tf.math.count_nonzero(output, dtype=tf.int32)
      self.assertAllEqual(ones_count, expected_ones)

  # @parameterized.parameters(
  #   ((30, 4),), ((1, 2, 1, 4),), ((3,3),)
  # )
  # def testMatrixDeterminism(self, shape):
  #   matrix1 = tf.ones(shape)
  #   matrix2 = tf.ones(shape)
  #   self.assertAllEqual(shape, matrix1.shape)
  #   self.assertAllEqual(shape, matrix2.shape)

  #   for ratio in RATIOS:
  #   # ratio = 0.4
  #     permuteones = sparse_utils.PermuteOnes(ratio)
  #     for seed in self.seeds:
  #       # seed = 5
  #       matrix1_sparse = permuteones(shape=shape, seed=seed)
  #       matrix2_sparse = permuteones(shape=shape, seed=seed)
  #       # print(matrix1_sparse, matrix2_sparse)
  #       self.assertAllClose(matrix1_sparse, matrix2_sparse, rtol=1.)
  #     del permuteones

if __name__ == '__main__':
  test.main()
