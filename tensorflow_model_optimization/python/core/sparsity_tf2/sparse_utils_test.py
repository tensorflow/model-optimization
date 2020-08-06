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
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity_tf2 import lthpruner as pruner
from tensorflow_model_optimization.python.core.sparsity_tf2 import sparse_utils

dtypes = tf.dtypes
test = tf.test


def sample_noise(x, mu=0, sigma=1.):
  sample = tf.random.normal((), mean=mu,  stddev=sigma, dtype=tf.float64)
  return sample

class SparseUtilsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(SparseUtilsTest, self).setUp()
    self.mask_init = lambda x: tf.ones(x)
    self.mask_init_w_type = lambda x, y: tf.ones(x, dtype=y)
    self.bernouilli = lambda p: sparse_utils.Bernouilli(p)
    self.permuteones = lambda ratio: sparse_utils.PermuteOnes(ratio)

  def testBernouilliMaskFraction(self):
    shape = (4, 4)
    mask = self.mask_init(shape)
    self.assertAllEqual(shape, mask.shape)

    seed = 0
    ratio = 0.5
    counts = []
    for _ in range(100):
      tmp = self.bernouilli(ratio)(mask.shape, seed=0)
      counts.append(np.count_nonzero(tmp))
    mean_count = np.mean(counts)
    self.assertAllEqual(mean_count, 4)

  def testDeterministicMaskFraction(self):
    # PermuteOnes
    shape = (4, 4)
    mask = self.mask_init(shape)
    self.assertAllEqual(shape, mask.shape)

    ratio = 0.5
    mask_sparse = self.permuteones(ratio)(mask.shape)
    self.assertAllEqual(np.count_nonzero(mask_sparse), 4)

  def testMaskDeterminism(self):
    shape = (4, 4)
    mask1 = self.mask_init(shape)
    mask2 = self.mask_init(shape)
    self.assertAllEqual(shape, mask1.shape)
    self.assertAllEqual(shape, mask2.shape)

    ratio = 0.5
    seed = 0
    permuteones = self.permuteones(ratio)
    mask1_sparse = permuteones(shape=mask1.shape, seed=seed) 
    mask2_sparse = permuteones(shape=mask2.shape, seed=seed)
    self.assertAllEqual(mask1_sparse, mask2_sparse)

  def testMaskDtype(self):
    dtypes = [tf.int32, tf.float32, tf.int64, tf.float64]
    shape = (3, 4)
    seed = 0
    type_checks = []
    for dtype in dtypes:
      mask = self.mask_init_w_type(shape, dtype)
      sparse_mask = self.permuteones(shape=mask.shape, dtype=dtype, seed=seed)
      type_checks.append(sparse_mask.dtype)
    self.assertAllEqual(dtypes, type_checks)

if __name__ == '__main__':
  test.main()
