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
  return np.sqrt(p * (1 - p))

class SparseUtilsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(SparseUtilsTest, self).setUp()
    self.ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    self.shapes = [(), [1], [5], [2, 2], [4, 4], [8, 4], [3, 17], [3, 6, 9]]
    self.seeds = range(20)

  def matrix_init(self, _shape):
    return tf.ones(_shape)
  
  def matrix_init_with_type(self, _shape, _type):
    tf.ones(_shape, dtype=_type)

  def get_bernouilli(self, p):
    return sparse_utils.Bernouilli(p)

  def get_permuteones(self, ratio):
    return sparse_utils.PermuteOnes(ratio)

  @parameters.parameters(
    (-1.,), (1.5,)
  )
  def testInvalidRatioRaisesError(self, p):
    with self.assertRaises(ValueError):
      bernouilli = self.get_bernouilli(p)
      permuteones = self.get_permuteones(p)

  @parameterized.parameters(
    (i,) for i in self.ratios
  )
  def testScatteredInitialOnes(self, ratio):
    shape = (4, 4)
    matrix = self.matrix_init(shape)
    self.assertAllEqual(shape, matrix.shape)

    def _check_scatter(sparse_matrix):
      ones_indices = np.where(sparse_matrix == 1.)
      mean_nonzeros = np.mean(ones_indices)
      ones_count = np.count_nonzero(sparse_matrix)
      return ones_count, mean_nonzeros

    seed = 0
    ones_indices = []
    ones_counts = []
    for _ in range(100):
      matrix_sparse_permute = self.get_permuteones(ratio)(shape, seed=seed)
      ones_count, mean_nonzeros = _check_scatter(matrix_sparse_permute)
      ones_indices.append(mean_nonzeros)
      ones_counts.append(ones_count)
      seed += 1
    # check that 1/0s are scattered
    # mean of entire matrix and mean of entire nonzeros
    self.assertAllClose(ones_indices[:len(ones_indices)//2], ones_indices[len(ones_indices)//2:])
    self.assertAllClose(ones_counts[:len(ones_counts)//2], ones_counts[len(ones_counts)//2:])

  @parameterized.parameters(
    (shape,) for shape in self.shapes
  )
  def testBernouilliMatrixFraction(self, shape):
    matrix = self.matrix_init(shape)
    self.assertAllEqual(shape, matrix.shape)

    seed = 0
    expected_counts = [16, 1.6, 4, 8, 2, 1.7, 0]
    expected_stdev = list(map(calculate_stdev, ratios))
    counts = []
    for ratio in self.ratios:
      ratio_list = []
      for _ in range(100):
        output = self.get_bernouilli(ratio)(shape, seed=seed)
        ratio_list.append(np.count_nonzero(output))
        self.assertAllEqual(output.shape, shape)
        seed += 1
      counts.append(ratio_list)
    mean_counts = np.mean(counts, axis=-1)
    stdev_counts = np.stdev(counts, axis=-1)
    self.assertAllClose(mean_counts, expected_counts)
    self.assertAllClose(stdev_counts, expected_stdev)

  @parameterized.parameters(
    (shape,) for shape in self.shapes
  )
  def testDeterministicmatrixFraction(self, shape):
    # PermuteOnes
    matrix = self.matrix_init(shape)
    self.assertAllEqual(shape, matrix.shape)

    for ratio in self.ratios:
      matrix_sparse = self.get_permuteones(ratio)(shape)
      self.assertAllEqual(np.count_nonzero(matrix_sparse), 8)

  @parameterized.parameters(
    (shape,) for shape in self.shapes
  )
  def testMatrixDeterminism(self):
    matrix1 = self.matrix_init(shape)
    matrix2 = self.matrix_init(shape)
    self.assertAllEqual(shape, matrix1.shape)
    self.assertAllEqual(shape, matrix2.shape)

    for ratio in self.ratios:
      permuteones = self.get_permuteones(ratio)
      for seed in self.seeds:
        matrix1_sparse = permuteones(shape=shape, seed=seed)
        matrix2_sparse = permuteones(shape=shape, seed=seed)
        self.assertAllEqual(matrix1_sparse, matrix2_sparse)

  @parameterized.parameters(
    (tf.int32,), (tf.float32,), (tf.int64,), (tf.float64,)
  )
  def testMatrixDtype(self, dtype):
    shape = (3, 4)
    seed = 0
    type_bern_checks = []
    typ_perm_checks = []
    matrix = self.matrix_init_w_type(shape, dtype)
    sparse_bernouilli = self.get_bernouilli(shape=matrix.shape, dtype=dtype, seed=seed)
    type_bern_checks.append(sparse_bernouilli.dtype)
    self.assertAllEqual(dtypes, type_bern_checks)
    sparse_permuteones = self.get_permuteones(shape=matrix.shape, dtype=dtype, seed=seed)
    type_perm_checks.append(sparse_permuteones.dtype)
    self.assertAllEqual(dtypes, type_perm_checks)

if __name__ == '__main__':
  test.main()
