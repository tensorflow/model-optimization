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
"""Tests for the key functions in pruner library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import g3

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# TODO(b/139939526): move to public API.
from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.sparsity_tf2 import pruner
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

dtypes = tf.dtypes
test = tf.test


def make_pruning_schedule(target_sparsity, begin, end, freq):
  return pruning_schedule.ConstantSparsity(target_sparsity, begin, end, freq)

def sample_noise(x, mu=0, sigma=1.):
  sample = tf.random.normal((), mean=mu,  stddev=sigma, dtype=tf.float64)
  return sample

def _dummy_gradient(x, dtype=tf.float32):
  try:
    base_type = x.dtype
  except:
    base_type = dtype
  grad = tf.ones_like(x, dtype=base_type)
  return grad

class PrunerTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(PrunerTest, self).setUp()
    self.block_size = (1, 1)
    self.block_pooling_type = "AVG"
    self.target_sparsity = 0.5
    self.constant_sparsity = pruning_schedule.ConstantSparsity(self.target_sparsity, 0, 100, 1)
    self.save_init = 0
    self.save_itr_10 = 10
    self.grad = _dummy_gradient

  # setUp() lies outside of the "eager scope" that wraps the test cases
  # themselves, resulting in initializing graph tensors instead of eager
  # tensors when testing eager execution.

  def testUpdateSingleMask(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100))
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        dtype=weight_dtype)
    threshold = tf.Variable(
        tf.zeros([], dtype=weight_dtype),  dtype=weight_dtype)
    pruning_vars = [(weight, mask, threshold)]

    p = pruner.LowMagnitudePruner(
        pruning_schedule=self.constant_sparsity,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    optimizer =  tf.keras.optimizers.SGD(learning_rate=0.01)
    optimizer.iterations.assign(0)
    p.create_slots(optimizer, weight)

    mask_before_pruning = optimizer.get_slot(weight, "mask").read_value()
    self.assertAllEqual(np.count_nonzero(mask_before_pruning), 100)

    next_step = optimizer.iterations.assign_add(1)
    p.update_masks(pruning_vars, next_step)

    mask_after_pruning = mask.read_value()
    self.assertAllEqual(np.count_nonzero(mask_after_pruning), 50)

  def testConstructsMaskAndThresholdCorrectly(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100))
    weight_dtype = weight.dtype.base_dtype
    p = pruner.LowMagnitudePruner(
        # Sparsity math often returns values with small tolerances.
        pruning_schedule=lambda x: (True, 0.200000018),
        block_size=(1, 1), block_pooling_type=None)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    optimizer.iterations.assign(0)
    p.create_slots(optimizer, weight)
    step = optimizer.iterations

    # input matrix is [ 1.0, 2.0, ..., 8.0, 9.0, 10.0 ]
    threshold, mask = p._update_mask(step, np.arange(1, 11))

    self.assertEqual(3, threshold)
    self.assertAllEqual(
        # expected matrix is [ 0.0, 0.0, 1.0, 1.0 ... 1.0 ]
        np.concatenate((np.zeros(2), np.ones(8))), mask)

  def _blockMasking(self, block_size, block_pooling_type, weight,
      expected_mask):
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight.dtype),
        dtype=weight.dtype)
    threshold = tf.Variable(
        tf.zeros([], dtype=weight.dtype),  dtype=weight.dtype)

    # Set up pruning
    p = pruner.LowMagnitudePruner(
        pruning_schedule=self.constant_sparsity,
        block_size=block_size,
        block_pooling_type=block_pooling_type)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    optimizer.iterations.assign(0)
    p.create_slots(optimizer, weight)
    step = optimizer.iterations

    _, new_mask = p._maybe_update_block_mask(step, weight)
    # Check if the mask is the same size as the weights
    self.assertAllEqual(new_mask.get_shape(), weight.get_shape())
    mask_after_pruning = new_mask
    self.assertAllEqual(mask_after_pruning, expected_mask)

  def testBlockMaskingAvg(self):
    block_size = (2, 2)
    block_pooling_type = "AVG"
    weight = tf.Variable([[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2],
                          [0.3, 0.3, 0.4, 0.4], [0.3, 0.3, 0.4, 0.4]])
    expected_mask = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                     [1., 1., 1., 1.], [1., 1., 1., 1.]]

    self._blockMasking(block_size, block_pooling_type, weight, expected_mask)

  def testBlockMaskingMax(self):
    block_size = (2, 2)
    block_pooling_type = "MAX"
    weight = tf.Variable([[0.1, 0.0, 0.2, 0.0], [0.0, -0.1, 0.0, -0.2],
                          [0.3, 0.0, 0.4, 0.0], [0.0, -0.3, 0.0,
                                                 -0.4]])
    expected_mask = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                     [1., 1., 1., 1.], [1., 1., 1., 1.]]

    self._blockMasking(block_size, block_pooling_type, weight, expected_mask)

  def testBlockMaskingWithHigherDimensionsRaisesError(self):
    block_size = (2, 2)
    block_pooling_type = "AVG"
    # Weights as in testBlockMasking, but with one extra dimension.
    weight = tf.Variable([[[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2],
                           [0.3, 0.3, 0.4, 0.4], [0.3, 0.3, 0.4,
                                                  0.4]]])
    expected_mask = [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                      [1., 1., 1., 1.], [1., 1., 1., 1.]]]

    # Block masking should only be used with 2 Dimensional weights.
    with self.assertRaises(ValueError):
      self._blockMasking(block_size, block_pooling_type, weight, expected_mask)

  def testConditionalMaskUpdate(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100))
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        dtype=weight_dtype)
    threshold = tf.Variable(
        tf.zeros([], dtype=weight_dtype), dtype=weight_dtype)
    pruning_vars = [(weight, mask, threshold)]

    def linear_sparsity(step):
      sparsity_val = tf.convert_to_tensor(
          [0.0, 0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5])
      return tf.convert_to_tensor(True), sparsity_val[step]

    def weight_mask_op(pruning_vars):
      values_and_vars = []
      for weight, mask, _ in  pruning_vars:
        weight.assign(tf.math.multiply(weight, mask))

    # Set up pruning
    p = pruner.LowMagnitudePruner(
        pruning_schedule=linear_sparsity,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    optimizer.iterations.assign(0)
    p.create_slots(optimizer, weight)
    step = optimizer.iterations

    non_zero_count = []
    for _ in range(10):
      step = optimizer.iterations
      p.update_masks(pruning_vars, step)
      weight_mask_op(pruning_vars)
      optimizer.iterations.assign_add(1)

      non_zero_count.append(tf.math.count_nonzero(weight))

    # Weights pruned at steps 1,3,5
    expected_non_zero_count = [100, 90, 90, 70, 70, 50, 50, 50, 50, 50]
    self.assertAllEqual(expected_non_zero_count, non_zero_count)


if __name__ == "__main__":
  test.main()
