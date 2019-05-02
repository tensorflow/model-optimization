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
"""Tests for the key functions in pruning library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import g3
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_impl
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils


@tf_test_util.run_all_in_graph_and_eager_modes
class PruningTest(test.TestCase):

  def setUp(self):
    super(PruningTest, self).setUp()
    self.global_step = K.zeros([], dtype=dtypes.int32)
    self.block_size = (1, 1)
    self.block_pooling_type = "AVG"

    def training_step_fn():
      return self.global_step

    self.constant_sparsity = pruning_schedule.ConstantSparsity(0.5, 0, 100, 1)
    self.training_step_fn = training_step_fn

  def testUpdateSingleMask(self):
    weight = K.variable(np.linspace(1.0, 100.0, 100), name="weights")
    mask = K.ones(weight.get_shape())
    threshold = K.zeros([])

    p = pruning_impl.Pruning(
        pruning_vars=[(weight, mask, threshold)],
        training_step_fn=self.training_step_fn,
        pruning_schedule=self.constant_sparsity,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    mask_before_pruning = K.get_value(mask)
    self.assertAllEqual(np.count_nonzero(mask_before_pruning), 100)

    if context.executing_eagerly():
      p.conditional_mask_update()
    else:
      K.get_session().run(p.conditional_mask_update())

    mask_after_pruning = K.get_value(mask)
    self.assertAllEqual(np.count_nonzero(mask_after_pruning), 50)

  def testConstructsMaskAndThresholdCorrectly(self):
    p = pruning_impl.Pruning(
        lambda: 0, None,
        # Sparsity math often returns values with small tolerances.
        lambda x: (True, 0.200000018),
        (1, 1), None)

    # input matrix is [ 1.0, 2.0, ..., 8.0, 9.0, 10.0 ]
    threshold, mask = p._update_mask(np.arange(1, 11))

    self.assertEqual(3, K.get_value(threshold))
    self.assertAllEqual(
        # expected matrix is [ 0.0, 0.0, 1.0, 1.0 ... 1.0 ]
        np.concatenate((np.zeros(2), np.ones(8))), K.get_value(mask))

  def _blockMasking(self, block_size, block_pooling_type, weight,
                    expected_mask):
    mask = K.ones(weight.get_shape())
    threshold = K.zeros([])

    # Set up pruning
    p = pruning_impl.Pruning(
        pruning_vars=[(weight, mask, threshold)],
        training_step_fn=self.training_step_fn,
        pruning_schedule=self.constant_sparsity,
        block_size=block_size,
        block_pooling_type=block_pooling_type)

    _, new_mask = p._maybe_update_block_mask(weight)
    # Check if the mask is the same size as the weights
    self.assertAllEqual(new_mask.get_shape(), weight.get_shape())
    mask_after_pruning = K.get_value(new_mask)
    self.assertAllEqual(mask_after_pruning, expected_mask)

  def testBlockMaskingAvg(self):
    block_size = (2, 2)
    block_pooling_type = "AVG"
    weight = K.variable([[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2],
                         [0.3, 0.3, 0.4, 0.4], [0.3, 0.3, 0.4, 0.4]])
    expected_mask = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                     [1., 1., 1., 1.], [1., 1., 1., 1.]]

    self._blockMasking(block_size, block_pooling_type, weight, expected_mask)

  def testBlockMaskingMax(self):
    block_size = (2, 2)
    block_pooling_type = "MAX"
    weight = constant_op.constant([[0.1, 0.0, 0.2, 0.0], [0.0, -0.1, 0.0, -0.2],
                                   [0.3, 0.0, 0.4, 0.0], [0.0, -0.3, 0.0,
                                                          -0.4]])
    expected_mask = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                     [1., 1., 1., 1.], [1., 1., 1., 1.]]

    self._blockMasking(block_size, block_pooling_type, weight, expected_mask)

  def testBlockMaskingWithHigherDimensionsRaisesError(self):
    block_size = (2, 2)
    block_pooling_type = "AVG"
    # Weights as in testBlockMasking, but with one extra dimension.
    weight = constant_op.constant([[[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2],
                                    [0.3, 0.3, 0.4, 0.4], [0.3, 0.3, 0.4,
                                                           0.4]]])
    expected_mask = [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                      [1., 1., 1., 1.], [1., 1., 1., 1.]]]

    # Block masking should only be used with 2 Dimensional weights.
    with self.assertRaises(ValueError):
      self._blockMasking(block_size, block_pooling_type, weight, expected_mask)

  def testConditionalMaskUpdate(self):
    weight = K.variable(np.linspace(1.0, 100.0, 100), name="weights")
    mask = K.ones(weight.get_shape())
    threshold = K.zeros([])

    def linear_sparsity(step):
      sparsity_val = ops.convert_to_tensor(
          [0.0, 0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5])
      return ops.convert_to_tensor(True), sparsity_val[step]

    # Set up pruning
    p = pruning_impl.Pruning(
        pruning_vars=[(weight, mask, threshold)],
        training_step_fn=self.training_step_fn,
        pruning_schedule=linear_sparsity,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    non_zero_count = []
    for _ in range(10):
      if context.executing_eagerly():
        p.conditional_mask_update()
        p.weight_mask_op()
        state_ops.assign_add(self.global_step, 1)
      else:
        K.get_session().run(p.conditional_mask_update())
        K.get_session().run(p.weight_mask_op())
        K.get_session().run(state_ops.assign_add(self.global_step, 1))

      non_zero_count.append(np.count_nonzero(K.get_value(weight)))

    # Weights pruned at steps 1,3,5
    expected_non_zero_count = [100, 90, 90, 70, 70, 50, 50, 50, 50, 50]
    self.assertAllEqual(expected_non_zero_count, non_zero_count)


if __name__ == "__main__":
  test.main()
