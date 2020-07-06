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
"""Tests for the key functions in lthpruner library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import g3

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
print("Running TF", tf.__version__)

# TODO(b/139939526): move to public API.
from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.keras import compat
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils
from tensorflow_model_optimization.python.core.sparsity_tf2 import lthpruner as pruner

dtypes = tf.dtypes
test = tf.test

def get_lth_sparsity(save_round, n_rounds, target_sparsity, end_epoch):
  """
  save_round: when to save the weight; 1 onwards
  n_rounds: number of pruning cycles to do
  target_sparsity: the sparsity percentage to achieve by the end of the iteration
  end_epoch: total number of epochs to train for where pruning is eligible

  Returns:
    percent to prune to after each cyle/round
  """
  # no pruning until weights are saved TODO: off by one??
  n_rounds = tf.constant(n_rounds, dtype='float32') # dtype='float32'
  frequency = tf.math.floordiv(end_epoch - save_round + 1, n_rounds) # range(0, end, freq)
  prune_ratio_per_round = tf.math.pow(target_sparsity, tf.math.divide(1, n_rounds))
  return tf.cast(frequency, tf.int64), prune_ratio_per_round

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

class PruningTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(PruningTest, self).setUp()
    self.block_size = (1, 1)
    self.block_pooling_type = "AVG"
    self.target_sparsity = 0.5
    self.constant_sparsity = pruning_schedule.ConstantSparsity(self.target_sparsity, 0, 100, 1)
    self.save_init = tf.Variable(0)
    self.save_itr_10 = 10
    self.dummy_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    self.grad = _dummy_gradient

  # Variable initialization outside of setUp() is needed for compatibility with
  # run_all_keras_modes.
  #
  # setUp() lies outside of the "eager scope" that wraps the test cases
  # themselves, resulting in initializing graph tensors instead of eager
  # tensors when testing eager execution.

  def testUpdateSingleMask(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        name="mask",
        dtype=weight_dtype)
    threshold = tf.Variable(
        tf.zeros([], dtype=weight_dtype), name="threshold", dtype=weight_dtype)
    pruning_vars = [(weight, mask, threshold)]

    p = pruner.LTHPruner(
        pruning_schedule=self.constant_sparsity,
        save_iteration=self.save_init,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    optimizer = self.dummy_optimizer
    optimizer.iterations.assign(0)
    p.create_slots(optimizer, weight)

    mask_before_pruning = optimizer.get_slot(weight, "mask").read_value()
    self.assertAllEqual(np.count_nonzero(mask_before_pruning), 100)

    next_step = optimizer.iterations.assign_add(1)
    p.update_masks(pruning_vars, next_step)

    mask_after_pruning = mask.read_value()
    self.assertAllEqual(np.count_nonzero(mask_after_pruning), 50)

  def testConstructsMaskAndThresholdCorrectly(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    weight_dtype = weight.dtype.base_dtype
    p = pruner.LTHPruner(
        # Sparsity math often returns values with small tolerances.
        pruning_schedule=lambda x: (True, 0.200000018),
        save_iteration=self.save_init,
        block_size=(1, 1), block_pooling_type=None)
        
    optimizer = self.dummy_optimizer
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
        name="mask",
        dtype=weight.dtype)
    threshold = tf.Variable(
        tf.zeros([], dtype=weight.dtype), name="threshold", dtype=weight.dtype)

    # Set up pruning
    p = pruner.LTHPruner(
        pruning_schedule=self.constant_sparsity,
        save_iteration=self.save_init,
        block_size=block_size,
        block_pooling_type=block_pooling_type)

    optimizer = self.dummy_optimizer
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
    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        name="mask",
        dtype=weight_dtype)
    threshold = tf.Variable(
        tf.zeros([], dtype=weight_dtype), name="threshold", dtype=weight_dtype)
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
    p = pruner.LTHPruner(
        pruning_schedule=linear_sparsity,
        save_iteration=self.save_init,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    optimizer = self.dummy_optimizer
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


  def testSaveOriginalInitializations(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    weight_dtype = weight.dtype.base_dtype
    save_round = 0
    n_rounds = 5
    end_iter = 100
    frequency, prune_ratio_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_iter)
    pruning_schedule = make_pruning_schedule(prune_ratio_per_round, 0, end_iter, frequency)
    
    p = pruner.LTHPruner(
        pruning_schedule=pruning_schedule,
        save_iteration=self.save_init,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    optimizer = self.dummy_optimizer
    optimizer.iterations.assign(0)
    init_weights_before_pruning = weight.read_value()

    def _train():
      p.create_slots(optimizer, weight)
      p.preprocess_weights(optimizer, weight, self.grad(weight))
      weight.assign(tf.math.add(weight, sample_noise(0))) # update weights
      p.postprocess_weights(optimizer, weight, self.grad(weight))
      optimizer.iterations.assign_add(1)
      initialization_slot = optimizer.get_slot(weight, "original_initialization")
      return initialization_slot
    
    initialization_slot = _train()
    self.assertAllEqual(initialization_slot, init_weights_before_pruning)
    
    initialization_slot = tf.function(_train)()
    self.assertAllEqual(initialization_slot, init_weights_before_pruning)


  def testSaveWeightsIterK(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")

    save_round = 5
    n_rounds = 24
    end_iter = 100
    frequency, prune_ratio_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_iter)
    pruning_schedule = make_pruning_schedule(1 - prune_ratio_per_round, save_round, end_iter, frequency)
    
    p = pruner.LTHPruner(
        pruning_schedule=pruning_schedule,
        save_iteration=save_round,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    optimizer = self.dummy_optimizer
    optimizer.iterations.assign(0)

    def _train(weight):
      expected_saved_initialization = tf.ones_like(weight, dtype=weight.dtype.base_dtype) * -1
      p.create_slots(optimizer, weight)
      for i in range(7):
        p.preprocess_weights(optimizer, weight, self.grad(weight))
        if optimizer.iterations == save_round:
          expected_saved_initialization = weight.read_value()
        weight.assign(tf.math.add(weight, sample_noise(i)))
        p.postprocess_weights(optimizer, weight, self.grad(weight))
        optimizer.iterations.assign_add(1)
      return expected_saved_initialization
      
    expected_saved_initialization = _train(weight)
    self.assertAllEqual(optimizer.get_slot(weight, "original_initialization"), expected_saved_initialization)

    initialization_slot_k = tf.math.multiply(optimizer.get_slot(weight, "original_initialization"), optimizer.get_slot(weight, "mask"))
    masked_expected = tf.math.multiply(expected_saved_initialization, optimizer.get_slot(weight, "mask"))
    self.assertAllEqual(initialization_slot_k, masked_expected)

    mask_after_pruning = optimizer.get_slot(weight, "mask").read_value()
    self.assertAllEqual(np.count_nonzero(mask_after_pruning), 97)

    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    optimizer.iterations.assign(0)
    expected_saved_initialization = tf.function(_train)(weight)
    self.assertAllEqual(optimizer.get_slot(weight, "original_initialization"), expected_saved_initialization)

    initialization_slot_k = tf.math.multiply(optimizer.get_slot(weight, "original_initialization"), optimizer.get_slot(weight, "mask"))
    masked_expected = tf.math.multiply(expected_saved_initialization, optimizer.get_slot(weight, "mask"))
    self.assertAllEqual(initialization_slot_k, masked_expected)

    mask_after_pruning = optimizer.get_slot(weight, "mask").read_value()
    self.assertAllEqual(np.count_nonzero(mask_after_pruning), 97)

  def testReloadWeightsatInitialization(self):
    weight1 = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")

    save_round = 0
    n_rounds = 20
    end_iter = 100
    frequency, prune_ratio_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_iter)
    pruning_schedule = make_pruning_schedule(1 - prune_ratio_per_round, save_round, end_iter, frequency)
    
    p = pruner.LTHPruner(
        pruning_schedule=pruning_schedule,
        save_iteration=self.save_init,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    optimizer = self.dummy_optimizer
    optimizer.iterations.assign(0)
    expected_saved_initialization = None

    def _train(weight):
      expected_saved_initialization = tf.ones_like(weight, dtype=weight.dtype.base_dtype) * -1
      p.create_slots(optimizer, weight)
      for i in tf.range(0, 2):
        p.preprocess_weights(optimizer, weight, self.grad(weight))
        if i == save_round:
          expected_saved_initialization = weight.read_value()
        weight.assign(tf.math.add(weight, sample_noise(i)))
        p.postprocess_weights(optimizer, weight, self.grad(weight))
        optimizer.iterations.assign_add(1) # save weights right before iteration
      return expected_saved_initialization

    expected_saved_initialization = _train(weight1)
    initialization_slot = optimizer.get_slot(weight1, "original_initialization")
    self.assertAllEqual(initialization_slot, expected_saved_initialization)

    mask_after_pruning = optimizer.get_slot(weight1, "mask").read_value()
    masked_weight_expected = tf.math.multiply(initialization_slot, mask_after_pruning)
    self.assertAllEqual(np.count_nonzero(masked_weight_expected), 97)
    self.assertAllEqual(np.count_nonzero(mask_after_pruning), 97)

    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    optimizer.iterations.assign(0)
    expected_saved_initialization = tf.function(_train)(weight)
    initialization_slot = optimizer.get_slot(weight, "original_initialization")
    self.assertAllEqual(initialization_slot, expected_saved_initialization)

    mask_after_pruning = optimizer.get_slot(weight, "mask").read_value()
    masked_weight_expected = tf.math.multiply(initialization_slot, mask_after_pruning)
    self.assertAllEqual(np.count_nonzero(masked_weight_expected), 97)
    self.assertAllEqual(np.count_nonzero(mask_after_pruning), 97)

  
  def testReloadWeightsIterationK(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")

    save_round = 5
    n_rounds = 24
    end_iter = 100
    frequency, prune_ratio_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_iter)
    pruning_schedule = make_pruning_schedule(1 - prune_ratio_per_round, save_round + 1, end_iter, frequency)
    
    p = pruner.LTHPruner(
        pruning_schedule=pruning_schedule,
        save_iteration=save_round,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    optimizer = self.dummy_optimizer
    optimizer.iterations.assign(0)

    def _train(weight):
      p.create_slots(optimizer, weight)
      for i in range(7):
        p.preprocess_weights(optimizer, weight, self.grad(weight))
        weight.assign(tf.math.add(weight, sample_noise(i)))
        p.postprocess_weights(optimizer, weight, self.grad(weight))
        optimizer.iterations.assign_add(1)

    _train(weight)
    masked_orig_init = tf.math.multiply(optimizer.get_slot(weight, "original_initialization"), optimizer.get_slot(weight, 'mask'))
    self.assertAllEqual(masked_orig_init, weight)

    mask_after_pruning = optimizer.get_slot(weight, "mask").read_value()
    self.assertAllEqual(np.count_nonzero(mask_after_pruning), 97)

    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    optimizer.iterations.assign(0)
    tf.function(_train)(weight)
    masked_orig_init = tf.math.multiply(optimizer.get_slot(weight, "original_initialization"), optimizer.get_slot(weight, 'mask'))
    self.assertAllEqual(masked_orig_init, weight)

    mask_after_pruning = optimizer.get_slot(weight, "mask").read_value()
    self.assertAllEqual(np.count_nonzero(mask_after_pruning), 97)


  def testReloadTwoTimes(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    weight_dtype = weight.dtype.base_dtype

    save_round = 5
    n_rounds = 24
    end_iter = 100
    frequency, prune_ratio_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_iter)
    pruning_schedule = make_pruning_schedule(1 - prune_ratio_per_round, save_round + 1, end_iter, frequency)
    
    p = pruner.LTHPruner(
        pruning_schedule=pruning_schedule,
        save_iteration=save_round,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    optimizer = self.dummy_optimizer
    optimizer.iterations.assign(0)

    def _train(weight):
      expected_first_saved_initialization = None
      expected_second_saved_initialization = None
      p.create_slots(optimizer, weight)
      for i in range(0, 13): # this should save, reload, and update exactly twice
        p.preprocess_weights(optimizer, weight, self.grad(weight))
        if i == save_round - 1:
          expected_first_saved_initialization = optimizer.get_slot(weight, "original_initialization")
        if i == (save_round - 1) * 2:
          expected_second_saved_initialization = optimizer.get_slot(weight, "original_initialization")
        weight.assign(tf.math.add(weight, sample_noise(i)))
        p.postprocess_weights(optimizer, weight, self.grad(weight))
        optimizer.iterations.assign_add(1)
      return expected_first_saved_initialization, expected_second_saved_initialization

    expected_first_saved_initialization, expected_second_saved_initialization = _train(weight)

    self.assertAllEqual(expected_first_saved_initialization, expected_second_saved_initialization)

    initialization_slot = optimizer.get_slot(weight, "original_initialization")
    self.assertAllEqual(initialization_slot, expected_first_saved_initialization)
    self.assertAllEqual(initialization_slot, expected_second_saved_initialization)

    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    optimizer.iterations.assign(0)
    expected_first_saved_initialization, expected_second_saved_initialization = tf.function(_train)(weight)

    self.assertAllEqual(expected_first_saved_initialization, expected_second_saved_initialization)

    initialization_slot = optimizer.get_slot(weight, "original_initialization")
    self.assertAllEqual(initialization_slot, expected_first_saved_initialization)
    self.assertAllEqual(initialization_slot, expected_second_saved_initialization)


if __name__ == "__main__":
  test.main()
