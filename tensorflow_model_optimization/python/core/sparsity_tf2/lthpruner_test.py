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
from tensorflow_model_optimization.python.core.keras import compat
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils
from tensorflow_model_optimization.python.core.sparsity_tf2 import lthpruner as pruner

K = tf.keras.backend
dtypes = tf.dtypes
test = tf.test


def assign_add(ref, value):
  if hasattr(tf, "assign_add"):
    return tf.assign_add(ref, value)
  else:
    return ref.assign_add(value)

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
  frequency = tf.math.floordiv(end_epoch - save_round + 1, n_rounds) # range(0, end, freq)
  prune_ratio_per_round = tf.math.pow(target_sparsity, tf.math.divide(1, n_rounds))
  return tf.cast(frequency, tf.int64), prune_ratio_per_round

def make_pruning_schedule(target_sparsity, begin, end, freq):
  return pruning_schedule.ConstantSparsity(target_sparsity, begin, end, freq)

def sample_noise(x, mu=0, sigma=1.):
  sample = tf.random.normal((), mean=mu,  stddev=sigma, dtype=tf.float64)
  return sample

def _dummy_gradient(x, dtype=tf.float64):
  try:
    base_type = x.dtype
  except:
    base_type = dtype
  return tf.ones_like(x, dtype=base_type)

class PruningTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(PruningTest, self).setUp()
    self.block_size = (1, 1)
    self.block_pooling_type = "AVG"
    self.target_sparsity = 0.5
    self.constant_sparsity = pruning_schedule.ConstantSparsity(self.target_sparsity, 0, 100, 1)
    self.save_init = tf.Variable(0)
    self.save_itr_10 = 10
    # dummy optimizer
    self.dummy_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    self.grad = _dummy_gradient

  # Variable initialization outside of setUp() is needed for compatibility with
  # run_all_keras_modes.
  #
  # setUp() lies outside of the "eager scope" that wraps the test cases
  # themselves, resulting in initializing graph tensors instead of eager
  # tensors when testing eager execution.
  def initialize(self):
    self.global_step = tf.Variable(
        tf.zeros([], dtype=dtypes.int32),
        dtype=dtypes.int32,
        # dtype=tf.float64,
        name="global_step")

    def training_step_fn():
      return self.global_step
    self.training_step_fn = training_step_fn

    compat.initialize_variables(self)

  # def testUpdateSingleMask(self):
  #   weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
  #   weight_dtype = weight.dtype.base_dtype
  #   mask = tf.Variable(
  #       tf.ones(weight.get_shape(), dtype=weight_dtype),
  #       name="mask",
  #       dtype=weight_dtype)
  #   threshold = tf.Variable(
  #       tf.zeros([], dtype=weight_dtype), name="threshold", dtype=weight_dtype)
  #   self.initialize()
  #   pruning_vars = [(weight, mask, threshold)]
  #   next_step = self.training_step_fn() + 1

  #   p = pruner.LTHPruner(
  #       pruning_schedule=self.constant_sparsity,
  #       save_iteration=self.save_init,
  #       block_size=self.block_size,
  #       block_pooling_type=self.block_pooling_type)

  #   mask_before_pruning = K.get_value(mask)
  #   self.assertAllEqual(np.count_nonzero(mask_before_pruning), 100)

  #   if tf.executing_eagerly():
  #     p.update_masks(pruning_vars, next_step)
  #   else:
  #     K.get_session().run(p.update_masks(pruning_vars, next_step))

  #   mask_after_pruning = K.get_value(mask)
  #   self.assertAllEqual(np.count_nonzero(mask_after_pruning), 50)

  # def testConstructsMaskAndThresholdCorrectly(self):
  #   self.initialize()
  #   p = pruner.LTHPruner(
  #       # Sparsity math often returns values with small tolerances.
  #       pruning_schedule=lambda x: (True, 0.200000018),
  #       save_iteration=self.save_init,
  #       block_size=(1, 1), block_pooling_type=None)
  #   step = self.global_step

  #   # input matrix is [ 1.0, 2.0, ..., 8.0, 9.0, 10.0 ]
  #   threshold, mask = p._update_mask(step, np.arange(1, 11))

  #   self.assertEqual(3, K.get_value(threshold))
  #   self.assertAllEqual(
  #       # expected matrix is [ 0.0, 0.0, 1.0, 1.0 ... 1.0 ]
  #       np.concatenate((np.zeros(2), np.ones(8))), K.get_value(mask))

  # def _blockMasking(self, block_size, block_pooling_type, weight,
  #                   expected_mask):
  #   mask = tf.Variable(
  #       tf.ones(weight.get_shape(), dtype=weight.dtype),
  #       name="mask",
  #       dtype=weight.dtype)
  #   threshold = tf.Variable(
  #       tf.zeros([], dtype=weight.dtype), name="threshold", dtype=weight.dtype)
  #   self.initialize()
  #   step = self.training_step_fn()

  #   # Set up pruning
  #   p = pruner.LTHPruner(
  #       pruning_schedule=self.constant_sparsity,
  #       save_iteration=self.save_init,
  #       block_size=block_size,
  #       block_pooling_type=block_pooling_type)

  #   _, new_mask = p._maybe_update_block_mask(step, weight)
  #   # Check if the mask is the same size as the weights
  #   self.assertAllEqual(new_mask.get_shape(), weight.get_shape())
  #   mask_after_pruning = K.get_value(new_mask)
  #   self.assertAllEqual(mask_after_pruning, expected_mask)

  # def testBlockMaskingAvg(self):
  #   block_size = (2, 2)
  #   block_pooling_type = "AVG"
  #   weight = tf.constant([[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2],
  #                         [0.3, 0.3, 0.4, 0.4], [0.3, 0.3, 0.4, 0.4]])
  #   expected_mask = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
  #                    [1., 1., 1., 1.], [1., 1., 1., 1.]]

  #   self._blockMasking(block_size, block_pooling_type, weight, expected_mask)

  # def testBlockMaskingMax(self):
  #   block_size = (2, 2)
  #   block_pooling_type = "MAX"
  #   weight = tf.constant([[0.1, 0.0, 0.2, 0.0], [0.0, -0.1, 0.0, -0.2],
  #                                  [0.3, 0.0, 0.4, 0.0], [0.0, -0.3, 0.0,
  #                                                         -0.4]])
  #   expected_mask = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
  #                    [1., 1., 1., 1.], [1., 1., 1., 1.]]

  #   self._blockMasking(block_size, block_pooling_type, weight, expected_mask)

  # def testBlockMaskingWithHigherDimensionsRaisesError(self):
  #   self.initialize()
  #   block_size = (2, 2)
  #   block_pooling_type = "AVG"
  #   # Weights as in testBlockMasking, but with one extra dimension.
  #   weight = tf.constant([[[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2],
  #                                   [0.3, 0.3, 0.4, 0.4], [0.3, 0.3, 0.4,
  #                                                          0.4]]])
  #   expected_mask = [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
  #                     [1., 1., 1., 1.], [1., 1., 1., 1.]]]

  #   # Block masking should only be used with 2 Dimensional weights.
  #   with self.assertRaises(ValueError):
  #     self._blockMasking(block_size, block_pooling_type, weight, expected_mask)

  # def testConditionalMaskUpdate(self):
  #   weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
  #   weight_dtype = weight.dtype.base_dtype
  #   mask = tf.Variable(
  #       tf.ones(weight.get_shape(), dtype=weight_dtype),
  #       name="mask",
  #       dtype=weight_dtype)
  #   threshold = tf.Variable(
  #       tf.zeros([], dtype=weight_dtype), name="threshold", dtype=weight_dtype)
  #   self.initialize()
  #   pruning_vars = [(weight, mask, threshold)]

  #   def linear_sparsity(step):
  #     sparsity_val = tf.convert_to_tensor(
  #         [0.0, 0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5])
  #     return tf.convert_to_tensor(True), sparsity_val[step]

  #   def weight_mask_op(pruning_vars):
  #     values_and_vars = []
  #     for weight, mask, _ in  pruning_vars:
  #       # values_and_vars.append((tf.math.multiply(weight, mask), weight))
  #       weight.assign(tf.math.multiply(weight, mask))
  #     # return tf.group(values_and_vars)

  #   # Set up pruning
  #   p = pruner.LTHPruner(
  #       pruning_schedule=linear_sparsity,
  #       save_iteration=self.save_init,
  #       block_size=self.block_size,
  #       block_pooling_type=self.block_pooling_type)
    
  #   step = self.training_step_fn

  #   non_zero_count = []
  #   for _ in range(10):
  #     if tf.executing_eagerly():
  #       p.update_masks(pruning_vars, step())
  #       weight_mask_op(pruning_vars)
  #       assign_add(self.global_step, 1)
  #     else:
  #       K.get_session().run(p.update_masks(pruning_vars, step()))
  #       K.get_session().run(weight_mask_op(pruning_vars))
  #       K.get_session().run(assign_add(self.global_step, 1))

  #     non_zero_count.append(np.count_nonzero(K.get_value(weight)))

  #   # Weights pruned at steps 1,3,5
  #   expected_non_zero_count = [100, 90, 90, 70, 70, 50, 50, 50, 50, 50]
  #   self.assertAllEqual(expected_non_zero_count, non_zero_count)

################################# LTH TESTS ###########################################
  # def testSaveOriginalInitializations(self):
  #   weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
  #   weight_dtype = weight.dtype.base_dtype
  #   save_round = 0
  #   n_rounds = 5
  #   end_iter = 100
  #   frequency, prune_ratio_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_iter)
  #   pruning_schedule = make_pruning_schedule(prune_ratio_per_round, 0, end_iter, frequency)

  #   self.initialize()
    
  #   p = pruner.LTHPruner(
  #       pruning_schedule=pruning_schedule,
  #       save_iteration=self.save_init,
  #       block_size=self.block_size,
  #       block_pooling_type=self.block_pooling_type)

  #   optimizer = self.dummy_optimizer
  #   optimizer.iterations.assign(0)
  #   init_weights_before_pruning = weight.read_value()

  #   p.create_slots(optimizer, weight)
  #   weight.assign(tf.math.add(weight, sample_noise(0)))
  #   p.prune(optimizer, weight, self.grad(weight))
  #   initialization_slot = optimizer.get_slot(weight, "original_initialization")
  #   self.assertAllEqual(initialization_slot, init_weights_before_pruning)


  # def testSaveWeightsIterK(self):
  #   weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
  #   weight_dtype = weight.dtype.base_dtype

  #   save_round = 5
  #   n_rounds = 24
  #   end_iter = 100
  #   frequency, prune_ratio_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_iter)
  #   # print(f"frequency {frequency} | ratio {prune_ratio_per_round}")
  #   pruning_schedule = make_pruning_schedule(1 - prune_ratio_per_round, save_round + 1, end_iter, frequency)

  #   self.initialize()
    
  #   p = pruner.LTHPruner(
  #       pruning_schedule=pruning_schedule,
  #       save_iteration=save_round,
  #       block_size=self.block_size,
  #       block_pooling_type=self.block_pooling_type)

  #   optimizer = self.dummy_optimizer
  #   optimizer.iterations.assign(0)
  #   expected_saved_initialization = None

  #   p.create_slots(optimizer, weight)
  #   for i in range(7): # this should save and mask update once, assumes pruning works correctly as per above tests
  #     # perturb weights
  #     weight.assign(tf.math.add(weight, sample_noise(i)))
  #     p.prune(optimizer, weight, self.grad(weight))
  #     # if i >= save_round:
  #     # if True:
  #       # print(f"prune iter {optimizer.iterations} og init {optimizer.get_slot(weight, 'original_initialization')}")
  #       # print(f"prune iter {optimizer.iterations} weights {weight}")
  #       # print(f"prune iter {optimizer.iterations} | mask {optimizer.get_slot(weight, 'mask')}")
  #     if save_round - 1 < 0 or i == save_round - 1:
  #       # print(f"hit round {optimizer.iterations} | expected {expected_saved_initialization}")
  #       expected_saved_initialization = weight.read_value()
  #       # print(f"iter {optimizer.iterations} | after expected {expected_saved_initialization}")
  #     optimizer.iterations.assign_add(1)

  #   # print("original init", optimizer.get_slot(weight, "original_initialization"))
  #   # print("expected init", expected_saved_initialization)
  #   self.assertAllEqual(optimizer.get_slot(weight, "original_initialization"), expected_saved_initialization)

  #   initialization_slot_k = tf.math.multiply(optimizer.get_slot(weight, "original_initialization"), optimizer.get_slot(weight, "mask"))
  #   masked_expected = tf.math.multiply(expected_saved_initialization, optimizer.get_slot(weight, "mask"))
  #   self.assertAllEqual(initialization_slot_k, masked_expected)

  #   mask_after_pruning = optimizer.get_slot(weight, "mask").read_value()
  #   # print(mask_after_pruning)
  #   self.assertAllEqual(np.count_nonzero(mask_after_pruning), 97)


  # def testReloadWeightsatInit(self): # TODO
  #   weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
  #   weight_dtype = weight.dtype.base_dtype

  #   save_round = 0
  #   n_rounds = 20
  #   end_iter = 100
  #   frequency, prune_ratio_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_iter)
  #   print(f"frequency {frequency} | ratio {prune_ratio_per_round}")
  #   pruning_schedule = make_pruning_schedule(1 - prune_ratio_per_round, save_round + 1, end_iter, frequency)

  #   self.initialize()
    
  #   p = pruner.LTHPruner(
  #       pruning_schedule=pruning_schedule,
  #       save_iteration=self.save_init,
  #       block_size=self.block_size,
  #       block_pooling_type=self.block_pooling_type)

  #   optimizer = self.dummy_optimizer
  #   optimizer.iterations.assign(0)
  #   expected_saved_initialization = None

  #   p.create_slots(optimizer, weight)
  #   for i in range(5): # this should save, reload, and update once
  #     weight.assign(tf.math.add(weight, sample_noise(i)))
  #     p.prune(optimizer, weight, self.grad(weight))
  #     print(f"prune iter {optimizer.iterations} og init {optimizer.get_slot(weight, 'original_initialization')}")
  #     print(f"prune iter {optimizer.iterations} weights {weight}")
  #     print(f"prune iter {optimizer.iterations} | mask {optimizer.get_slot(weight, 'mask')}")
  #     if i == save_round:
  #       expected_saved_initialization = weight.read_value()
  #       print(f"iter {optimizer.iterations} | after expected {expected_saved_initialization}")
  #     optimizer.iterations.assign_add(1) # save weights right before iteration

  #   initialization_slot = optimizer.get_slot(weight, "original_initialization")
  #   print("init slooooot", initialization_slot)
  #   self.assertAllEqual(initialization_slot, expected_saved_initialization)

    # mask_after_pruning = optimizer.get_slot(weight, "mask").read_value()
    # self.assertAllEqual(np.count_nonzero(mask_after_pruning), 1 - tf.math.multiply(tf.math.pow(0.5, tf.math.divide(1, n_rounds)), 100))

    # masked_weights_after_pruning = tf.multiply(optimizer.get_slot(weight, "mask"), initialization_slot)
    # self.assertAllEqual(masked_weights_after_pruning, weight)

  # def testReloadAfterSaveInit(self): # TODO
  #   weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
  #   print("initial", weight)
  #   weight_dtype = weight.dtype.base_dtype
  #   print('weight_dtype', weight_dtype)

  #   save_round = 0
  #   n_rounds = 20
  #   end_iter = 100
  #   frequency, prune_ratio_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_iter)
  #   pruning_schedule = make_pruning_schedule(prune_ratio_per_round, save_round, end_iter, frequency)
  #   # NOTE: save_round > start = 0, should not prune until save round is passed

  #   self.initialize()
    
  #   p = pruner.LTHPruner(
  #       pruning_schedule=pruning_schedule,
  #       save_iteration=self.save_init,
  #       block_size=self.block_size,
  #       block_pooling_type=self.block_pooling_type)

  #   optimizer = self.dummy_optimizer
  #   optimizer.iterations.assign(tf.cast(self.training_step_fn(), tf.int64))
  #   expected_saved_initialization = None

  #   p.create_slots(optimizer, weight)
  #   for i in range(5): # this should save, reload, and update once
  #     weight = tf.math.add(weight, sample_noise(i))
  #     p.prune(optimizer, weight, self.grad(weight))
  #     if i == save_round - 1: # TODO: minus 1???
  #       expected_saved_initialization = weight.read_value()
  #     optimizer.iterations.assign(tf.Variable(i))

  #   initialization_slot = optimizer.get_slot(weight, "original_initialization")
  #   self.assertAllEqual(initialization_slot, expected_saved_initialization)

  #   mask_after_pruning = optimizer.get_slot(weight, "mask").read_value()
  #   self.assertAllEqual(np.count_nonzero(mask_after_pruning), tf.math.multiply(tf.math.pow(0.5, tf.math.divide(1, n_rounds)), 100))

  #   masked_weights_after_pruning = tf.multiply(optimizer.get_slot(weight, "mask"), initialization_slot)
  #   self.assertAllEqual(masked_weights_after_pruning, weight)
  
  def testReloadWeightsIterationK(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    weight_dtype = weight.dtype.base_dtype

    save_round = 5
    n_rounds = 24
    end_iter = 100
    frequency, prune_ratio_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_iter)
    pruning_schedule = make_pruning_schedule(1 - prune_ratio_per_round, save_round + 1, end_iter, frequency)

    self.initialize()
    
    p = pruner.LTHPruner(
        pruning_schedule=pruning_schedule,
        save_iteration=save_round,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    optimizer = self.dummy_optimizer
    optimizer.iterations.assign(0)

    p.create_slots(optimizer, weight)
    for i in range(7): # this should save and mask update once, assumes pruning works correctly as per above tests
      weight.assign(tf.math.add(weight, sample_noise(i)))
      p.prune(optimizer, weight, self.grad(weight))
      optimizer.iterations.assign_add(1)

    masked_orig_init = tf.math.multiply(optimizer.get_slot(weight, "original_initialization"), optimizer.get_slot(weight, 'mask'))
    self.assertAllEqual(masked_orig_init, weight)

    mask_after_pruning = optimizer.get_slot(weight, "mask").read_value()
    self.assertAllEqual(np.count_nonzero(mask_after_pruning), 97)


  # def testReloadTwoTimes(self):
  #   weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
  #   weight_dtype = weight.dtype.base_dtype

  #   save_round = 5
  #   n_rounds = 24
  #   end_iter = 100
  #   frequency, prune_ratio_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_iter)
  #   pruning_schedule = make_pruning_schedule(prune_ratio_per_round, save_round, end_iter, frequency)
  #   # NOTE: save_round > start = 0, should not prune until save round is passed

  #   self.initialize()
    
  #   p = pruner.LTHPruner(
  #       pruning_schedule=pruning_schedule,
  #       save_iteration=self.save_init,
  #       block_size=self.block_size,
  #       block_pooling_type=self.block_pooling_type)

  #   training_step = 0
  #   optimizer = self.dummy_optimizer
  #   optimizer.iterations.assign(tf.cast(self.training_step_fn(), tf.int64))
  #   expected_first_saved_initialization = None
  #   expected_second_saved_initialization = None

  #   reload_itrs = []
  #   p.create_slots(optimizer, weight)
  #   for i in range(0, 13): # this should save, reload, and update exactly twice
  #     weight = tf.math.add(weight, sample_noise(i))
  #     p.prune(optimizer, weight, self.grad(weight))
  #     if i == save_round:
  #       expected_first_saved_initialization = optimizer.get_slot(weight, "original_initialization")
  #     if i == save_round * 2:
  #       expected_second_saved_initialization = optimizer.get_slot(weight, "original_initialization")
  #     optimizer.iterations.assign(tf.Variable(i)) # assign the optimizer step

  #   self.assertAllEqual(expected_first_saved_initialization, expected_second_saved_initialization)

  #   initialization_slot = optimizer.get_slot(weight, "original_initialization")
  #   self.assertAllEqual(initialization_slot, expected_first_saved_initialization)
  #   self.assertAllEqual(initialization_slot, expected_second_saved_initialization)


if __name__ == "__main__":
  test.main()
