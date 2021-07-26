# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for utility functions in pruning_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import g3
from absl.testing import parameterized

import tensorflow as tf
from tensorflow_model_optimization.python.core.keras import compat
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils

glorot_uniform_initializer = tf.keras.initializers.glorot_uniform


@parameterized.named_parameters(
    ("1x1", [1, 1]), ("4x4", [4, 4]), ("6x6", [6, 6]), ("1x4", [1, 4]),
    ("4x1", [4, 1]), ("1x8", [1, 8]), ("8x1", [8, 1]))
class PruningUtilsParameterizedTest(tf.test.TestCase, parameterized.TestCase):

  def _compare_pooling_methods(self, weights, pooling_kwargs):
    with self.cached_session():
      compat.initialize_variables(self)
      pooled_weights_tf = tf.squeeze(
          tf.nn.pool(
              tf.reshape(
                  weights,
                  [1, weights.get_shape()[0],
                   weights.get_shape()[1], 1]), **pooling_kwargs))
      pooled_weights_factorized_pool = pruning_utils.factorized_pool(
          weights, **pooling_kwargs)
      self.assertAllClose(self.evaluate(pooled_weights_tf),
                          self.evaluate(pooled_weights_factorized_pool))

  def _compare_expand_tensor_with_kronecker_product(self, tensor, block_dim):
    with self.cached_session() as session:
      compat.initialize_variables(self)
      expanded_tensor = pruning_utils.expand_tensor(tensor, block_dim)
      kronecker_product = pruning_utils.kronecker_product(
          tensor, tf.ones(block_dim))
      expanded_tensor_val, kronecker_product_val = session.run(
          [expanded_tensor, kronecker_product])
      self.assertAllEqual(expanded_tensor_val, kronecker_product_val)

  def testFactorizedAvgPool(self, window_shape):
    shape = [1024, 2048]
    weights = tf.Variable(
        glorot_uniform_initializer()(shape), shape=shape, name="weights")
    pooling_kwargs = {
        "window_shape": window_shape,
        "pooling_type": "AVG",
        "strides": window_shape,
        "padding": "SAME"
    }
    self._compare_pooling_methods(weights, pooling_kwargs)

  def testFactorizedMaxPool(self, window_shape):
    shape = [1024, 2048]
    weights = tf.Variable(
        glorot_uniform_initializer()(shape), shape=shape, name="weights")
    pooling_kwargs = {
        "window_shape": window_shape,
        "pooling_type": "MAX",
        "strides": window_shape,
        "padding": "SAME"
    }
    self._compare_pooling_methods(weights, pooling_kwargs)

  def testExpandTensor(self, block_dim):
    weights = tf.random.normal(shape=[1024, 512])
    self._compare_expand_tensor_with_kronecker_product(weights, block_dim)


class GenerateMbyNMaskTest(tf.test.TestCase, parameterized.TestCase):
  @parameterized.named_parameters(
      {
          "testcase_name": "wt_2x1_mbyn_2x4",
          "weights": [[1.24], [1.11]],
          "m_by_n": (2, 4),
          "expected_mask": [[1], [1]],
      },
      {
            "testcase_name": "wt_2x3_mbyn_2x4",
            "weights": [[-0.49, -0.66, 0.42], [0.31, 0.63, -1.63]],
            "m_by_n": (2, 4),
            "expected_mask": [[1, 1, 0], [0, 1, 1]],
      },
      {
            "testcase_name": "wt_2x3_mbyn_1x2",
            "weights": [[-0.49, -0.66, 0.42], [0.31, 0.63, -1.63]],
            "m_by_n": (1, 2),
            "expected_mask": [[0, 1, 1], [0, 1, 1]],
      },
      {
          "testcase_name": "wt_2x4_mbyn_2x4",
          "weights": [[0.25, -1.33, 0.43, -0.11], [0.47, 0.92, -0.13, -2.23]],
          "m_by_n": (2, 4),
          "expected_mask": [[0, 1, 1, 0], [0, 1, 0, 1]],
      },
      {
          "testcase_name": "wt_2x5_mbyn_3x5",
          "weights": [
              [1.71, -0.27, -0.3, -0.61, -0.19],
              [-0.4, 1.12, 0.46, -1.12, 0.17]
          ],
          "m_by_n": (3, 5),
          "expected_mask": [[1, 0, 0, 1, 0], [0, 1, 0, 1, 0]],
      },
  )
  def testGenerateMbyNMask(self, weights, m_by_n, expected_mask):
    mask = pruning_utils.generate_m_by_n_mask(weights, m_by_n)
    self.assertAllEqual(mask, expected_mask)

  @parameterized.parameters(
    {"m_by_n": (4, 4)},
    {"m_by_n": (5, 4)},
  )
  def testGenerateMbyNMaskRaise(self, m_by_n):
    shape = [2, 5]
    weights = tf.Variable(
        glorot_uniform_initializer()(shape), shape=shape, name="weights")
    with self.assertRaises(tf.errors.InvalidArgumentError):
      pruning_utils.generate_m_by_n_mask(weights, m_by_n)


class IsPrunedMbyNTest(tf.test.TestCase, parameterized.TestCase):
  @parameterized.parameters(
    ([[1, 2, 0, 0]], "C_OUT", True),
    ([[1, 2, 0, 0]], "C_OUT", False, (1, 2)),
    ([[1, 0, 2, 0]], "C_OUT", True, (1, 2)),
    ([[1, 2, 3, 4]], "C_OUT", False),
    ([[0, 0, 0, 0]], "C_OUT", True),
    ([[1, 2, 0, 4], [0, 0, 1, 2]], "C_OUT", False),
    ([[1, 2, 0, 0], [0, 0, 0, 1], [0, 2, 1, 0]], "C_OUT", True),
    ([[1, 2, 0, 0], [1, 0, 0, 1], [0, 2, 1, 0]], "C_IN", True),
    ([[1, 2, 3, 4, 5, 6]], "C_OUT", False),
    ([[1, 2, 0, 0, 5, 6]], "C_OUT", True),
    ([[1, 2, 0, 0, 5, 6, 7]], "C_OUT", False),
    ([[1, 2, 0, 0, 5, 6, 0]], "C_OUT", True),
  )
  def testIsPruned2by4(self, weight, last_channel, expected, m_by_n=(2, 4)):
    if last_channel == "C_OUT":
      weight = tf.transpose(tf.constant(weight))
    elif last_channel == "C_IN":
      weight = tf.constant(weight)
    answer = pruning_utils.is_pruned_m_by_n(weight, m_by_n, last_channel)
    self.assertEqual(answer, expected)

class WeightsRearrangeTest(tf.test.TestCase, parameterized.TestCase):
  @parameterized.parameters(
    {"shape": [4,3], "expected_shape": [3,4]},
    {"shape": [5,5,3,4], "expected_shape": [100,3]}
  )
  def testWeightsRearrange(self, shape, expected_shape):
    weights = tf.Variable(
        glorot_uniform_initializer()(shape), shape=shape, name="weights")
    prepared_wt = pruning_utils.weights_rearrange(weights)
    self.assertEqual(prepared_wt.shape, expected_shape)

  @parameterized.parameters(
    {"shape": [2]},
    {"shape": [2,2,2]}
  )
  def testWeightsRearrangeRaise(self, shape):
    weights = tf.Variable(
        glorot_uniform_initializer()(shape), shape=shape, name="weights")
    with self.assertRaises(ValueError):
      pruning_utils.weights_rearrange(weights)


class MbyNSparsityMaskPrepareTest(tf.test.TestCase, parameterized.TestCase):
  @parameterized.parameters(
    {"mask_shape": [4,4], "weights_shape": [2,2,4,1]},
    {"mask_shape": [2,4], "weights_shape": [4,2]},
  )
  def testMbyNSparsityMaskPrepare(self, mask_shape, weights_shape):
    mask= tf.ones(shape=mask_shape, name="mask")
    shape = tf.TensorShape(weights_shape)
    pruning_utils.m_by_n_sparsity_mask_prepare(mask, shape)

  @parameterized.parameters(
      {
          "mask_shape": [2, 2],
          "weights_shape": [2, 1],
          "error_type": tf.errors.InvalidArgumentError,
      },
      {"mask_shape": [4, 4, 1], "weights_shape": [2, 2, 4, 1], "error_type": ValueError},
      {"mask_shape": [4, 4], "weights_shape": [2, 2, 4], "error_type": ValueError},
  )
  def testMbyNSparsityMaskPrepareRaise(self, mask_shape, weights_shape, error_type):
    mask = tf.ones(shape=mask_shape, name="mask")
    shape = tf.TensorShape(weights_shape)
    with self.assertRaises(error_type):
      pruning_utils.m_by_n_sparsity_mask_prepare(mask, shape)

if __name__ == "__main__":
  tf.test.main()
