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
from tensorflow_model_optimization.python.core.sparsity_tf2 import pruner
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils

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
  prune_pcent_per_round = tf.math.pow(target_sparsity, tf.math.divide(1, n_rounds))
  return tf.cast(frequency, tf.int64), prune_pcent_per_round

def make_pruning_schedule(target_sparsity, begin, end, freq):
  return pruning_schedule.ConstantSparsity(target_sparsity, begin, end, freq)

def sample_noise(x, mu=0, sigma=1.):
  dist = tf.random.normal(loc=mu,  sccale=sigma)
  return dist.prob(x)

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
    self.grad = lambda shape: tf.ones_like(shape)

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
  #       save_schedule=self.save_init,
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
  #       save_schedule=self.save_init,
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
  #       save_schedule=self.save_init,
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
  #       save_schedule=self.save_init,
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


  def testSaveOGSlotInit(self):
    initial_weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    weight_dtype = initial_weight.dtype.base_dtype
    save_round = 0
    n_rounds = 5
    end_itr = 100
    frequency, prune_pcent_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_itr)
    pruning_schedule = make_pruning_schedule(prune_pcent_per_round, 0, end_itr, frequency)

    self.initialize()
    
    p = pruner.LTHPruner(
        pruning_schedule=pruning_schedule,
        save_schedule=self.save_init,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    optimizer = self.dummy_optimizer
    optimizer.iterations.assign(tf.cast(self.training_step_fn(), tf.int64))
    init_weights_before_pruning = initial_weight.read_value()

    # TODO: before creating slots, should exist
    p.create_slots(optimizer, initial_weight)
    initial_weight.assign(tf.math.add(initial_weight, sample_noise(0)))
    p.prune(optimizer, initial_weight, self.grad(initial_weight))
    og_initialization_slot = optimizer.get_slot(initial_weight, "original_initialization")
    self.assertAllEqual(og_initialization_slot, init_weights_before_pruning)


  # def testSaveWeightsItrK(self):
  #   initial_weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
  #   weight_dtype = initial_weight.dtype.base_dtype

  #   save_round = 5
  #   n_rounds = 20
  #   end_itr = 100
  #   frequency, prune_pcent_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_itr)
  #   pruning_schedule = make_pruning_schedule(prune_pcent_per_round, save_round, end_itr, frequency)
  #   # NOTE: save_round > start = 0, should not prune until save round is passed

  #   self.initialize()
    
  #   p = pruner.LTHPruner(
  #       pruning_schedule=pruning_schedule,
  #       save_schedule=self.save_init,
  #       block_size=self.block_size,
  #       block_pooling_type=self.block_pooling_type)

  #   optimizer = self.dummy_optimizer
  #   optimizer.iterations.assign(tf.cast(self.training_step_fn(), tf.int64))
  #   expected_saved_og_initialization = None

  #   p.create_slots(optimizer, initial_weight)
  #   for i in range(9): # this should save and not update, assumes pruning works correctly as per above tests
  #     # perturb weights
  #     initial_weight = tf.math.add(initial_weight, sample_noise(i))
  #     p.prune(optimizer, initial_weight, self.grad(initial_weight))
  #     if i == save_round:
  #       expected_saved_og_initialization = optimizer.get_slot(initial_weight, "original_initialization").read_value()
  #     optimizer.iterations.assign(tf.Variable(i))

  #   og_initialization_slot = optimizer.get_slot(initial_weight, "original_initialization")
  #   self.assertAllEqual(og_initialization_slot, expected_saved_og_initialization)

  #   mask_after_pruning = K.get_value(optimizer.get_slot(initial_weight, "mask"))
  #   self.assertAllEqual(np.count_nonzero(mask_after_pruning), 100)


  # def testReloadWeightsatInit(self):
  #   initial_weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
  #   weight_dtype = initial_weight.dtype.base_dtype

  #   save_round = 0
  #   n_rounds = 20
  #   end_itr = 100
  #   frequency, prune_pcent_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_itr)
  #   pruning_schedule = make_pruning_schedule(prune_pcent_per_round, save_round, end_itr, frequency)
  #   # NOTE: save_round > start = 0, should not prune until save round is passed

  #   self.initialize()
    
  #   p = pruner.LTHPruner(
  #       pruning_schedule=pruning_schedule,
  #       save_schedule=self.save_init,
  #       block_size=self.block_size,
  #       block_pooling_type=self.block_pooling_type)

  #   optimizer = self.dummy_optimizer
  #   optimizer.iterations.assign(tf.cast(self.training_step_fn(), tf.int64))
  #   expected_saved_og_initialization = None

  #   p.create_slots(optimizer, initial_weight)
  #   for i in range(5): # this should save, reload, and update once
  #     initial_weight = tf.math.add(initial_weight, sample_noise(i))
  #     p.prune(optimizer, initial_weight, self.grad(initial_weight))
  #     if i == save_round:
  #       expected_saved_og_initialization = optimizer.get_slot(initial_weight, "original_initialization")
  #     optimizer.iterations.assign(tf.Variable(i)) # save weights right before iteration

  #   og_initialization_slot = optimizer.get_slot(initial_weight, "original_initialization")
  #   self.assertAllEqual(og_initialization_slot, expected_saved_og_initialization)

  #   mask_after_pruning = optimizer.get_slot(initial_weight, "mask").read_value()
  #   self.assertAllEqual(np.count_nonzero(mask_after_pruning), tf.math.multiply(tf.math.pow(0.5, tf.math.divide(1, n_rounds)), 100))

  #   masked_weights_after_pruning = tf.multiply(optimizer.get_slot(initial_weight, "mask"), og_initialization_slot)
  #   self.assertAllEqual(masked_weights_after_pruning, initial_weight)

  # def testReloadAfterSaveInit(self):
  #   initial_weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
  #   print("initial", initial_weight)
  #   weight_dtype = initial_weight.dtype.base_dtype
  #   print('weight_dtype', weight_dtype)

  #   save_round = 0
  #   n_rounds = 20
  #   end_itr = 100
  #   frequency, prune_pcent_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_itr)
  #   pruning_schedule = make_pruning_schedule(prune_pcent_per_round, save_round, end_itr, frequency)
  #   # NOTE: save_round > start = 0, should not prune until save round is passed

  #   self.initialize()
    
  #   p = pruner.LTHPruner(
  #       pruning_schedule=pruning_schedule,
  #       save_schedule=self.save_init,
  #       block_size=self.block_size,
  #       block_pooling_type=self.block_pooling_type)

  #   optimizer = self.dummy_optimizer
  #   optimizer.iterations.assign(tf.cast(self.training_step_fn(), tf.int64))
  #   expected_saved_og_initialization = None

  #   p.create_slots(optimizer, initial_weight)
  #   for i in range(5): # this should save, reload, and update once
  #     initial_weight = tf.math.add(initial_weight, sample_noise(i))
  #     p.prune(optimizer, initial_weight, self.grad(initial_weight))
  #     if i == save_round - 1: # TODO: minus 1???
  #       expected_saved_og_initialization = optimizer.get_slot(initial_weight, "original_initialization")
  #     optimizer.iterations.assign(tf.Variable(i))

  #   og_initialization_slot = optimizer.get_slot(initial_weight, "original_initialization")
  #   self.assertAllEqual(og_initialization_slot, expected_saved_og_initialization)

  #   mask_after_pruning = optimizer.get_slot(initial_weight, "mask").read_value()
  #   self.assertAllEqual(np.count_nonzero(mask_after_pruning), tf.math.multiply(tf.math.pow(0.5, tf.math.divide(1, n_rounds)), 100))

  #   masked_weights_after_pruning = tf.multiply(optimizer.get_slot(initial_weight, "mask"), og_initialization_slot)
  #   self.assertAllEqual(masked_weights_after_pruning, initial_weight)
  
  # def testReloadWeightsIterK(self):
  #   initial_weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
  #   weight_dtype = initial_weight.dtype.base_dtype

  #   save_round = 5
  #   n_rounds = 24
  #   end_itr = 100
  #   frequency, prune_pcent_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_itr)
  #   pruning_schedule = make_pruning_schedule(prune_pcent_per_round, save_round, end_itr, frequency)
  #   # NOTE: save_round > start = 0, should not prune until save round is passed

  #   self.initialize()
    
  #   p = pruner.LTHPruner(
  #       pruning_schedule=pruning_schedule,
  #       save_schedule=self.save_init,
  #       block_size=self.block_size,
  #       block_pooling_type=self.block_pooling_type)

  #   optimizer = self.dummy_optimizer
  #   optimizer.iterations.assign(tf.cast(self.training_step_fn(), tf.int64))
  #   expected_saved_og_initialization = None

  #   p.create_slots(optimizer, initial_weight)
  #   for i in range(9): # this should save, reload, and update once
  #     initial_weight = tf.math.add(initial_weight, sample_noise(i))
  #     p.prune(optimizer, initial_weight, self.grad(initial_weight))
  #     if i == save_round:
  #       expected_saved_og_initialization = optimizer.get_slot(initial_weight, "original_initialization")
  #     optimizer.iterations.assign(tf.Variable(i))

  #   og_initialization_slot = optimizer.get_slot(initial_weight, "original_initialization")
  #   self.assertAllEqual(og_initialization_slot, expected_saved_og_initialization)

  #   mask_after_pruning = K.get_value(optimizer.get_slot(initial_weight, "mask"))
  #   self.assertAllEqual(np.count_nonzero(mask_after_pruning), tf.math.pow(100, tf.math.divide(1, n_rounds)))

  #   masked_weights_after_pruning = tf.multiply(optimizer.get_slot(initial_weight, "mask"), og_initialization_slot)
  #   self.assertAllEqual(masked_weights_after_pruning, initial_weight)


  # def testReloadTwoTimes(self):
  #   initial_weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
  #   weight_dtype = initial_weight.dtype.base_dtype

  #   save_round = 5
  #   n_rounds = 24
  #   end_itr = 100
  #   frequency, prune_pcent_per_round = get_lth_sparsity(save_round, n_rounds, self.target_sparsity, end_itr)
  #   pruning_schedule = make_pruning_schedule(prune_pcent_per_round, save_round, end_itr, frequency)
  #   # NOTE: save_round > start = 0, should not prune until save round is passed

  #   self.initialize()
    
  #   p = pruner.LTHPruner(
  #       pruning_schedule=pruning_schedule,
  #       save_schedule=self.save_init,
  #       block_size=self.block_size,
  #       block_pooling_type=self.block_pooling_type)

  #   training_step = 0
  #   optimizer = self.dummy_optimizer
  #   optimizer.iterations.assign(tf.cast(self.training_step_fn(), tf.int64))
  #   expected_first_saved_og_initialization = None
  #   expected_second_saved_og_initialization = None

  #   reload_itrs = []
  #   p.create_slots(optimizer, initial_weight)
  #   for i in range(0, 13): # this should save, reload, and update exactly twice
  #     initial_weight = tf.math.add(initial_weight, sample_noise(i))
  #     p.prune(optimizer, initial_weight, self.grad(initial_weight))
  #     if i == save_round:
  #       expected_first_saved_og_initialization = optimizer.get_slot(initial_weight, "original_initialization")
  #     if i == save_round * 2:
  #       expected_second_saved_og_initialization = optimizer.get_slot(initial_weight, "original_initialization")
  #     optimizer.iterations.assign(tf.Variable(i)) # assign the optimizer step

  #   self.assertAllEqual(expected_first_saved_og_initialization, expected_second_saved_og_initialization)

  #   og_initialization_slot = optimizer.get_slot(initial_weight, "original_initialization")
  #   self.assertAllEqual(og_initialization_slot, expected_first_saved_og_initialization)
  #   self.assertAllEqual(og_initialization_slot, expected_second_saved_og_initialization)


if __name__ == "__main__":
  test.main()