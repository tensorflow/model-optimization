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
"""Tests for the key functions in riglpruner library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import g3

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# TODO(b/139939526): move to public API.
from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.sparsity_tf2 import schedule
from tensorflow_model_optimization.python.core.sparsity_tf2 import riglpruner as pruner

dtypes = tf.dtypes
test = tf.test


def make_update_schedule(fraction, begin, end, freq):
  return schedule.ConstantSparsity(fraction, begin, end, freq)

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

class RiglPruningTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RiglPruningTest, self).setUp()
    self.block_size = (1, 1)
    self.block_pooling_type = "AVG"
    self.target_sparsity = 0.5
    self.initial_drop_fraction = 0.3 # i.e. 0.3 * target sparsity per layer, annealed by schedule below
    self.constant_updater = schedule.ConstantSparsity(self.initial_drop_fraction, 0, 100, 1)
    self.skip_updater = schedule.ConstantSparsity(self.initial_drop_fraction, 0, 100, 2)
    self.skip_update = sccche
    self.grad = _dummy_gradient
    self.seed = 0
    self.noise_std = 1
    self.reinit = False
    self.stateless = True

  # setUp() lies outside of the "eager scope" that wraps the test cases
  # themselves, resulting in initializing graph tensors instead of eager
  # tensors when testing eager execution.

  def testMaskNoChangeBeforeandAfter(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100))
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        dtype=weight_dtype)
    sparse_vars = [(mask, weight, self.grad(weight))]

    p = pruner.RiGLPruner(
      update_schedule=self.skip_updater,
      sparsity=self.target_sparsity,
      block_size=self.block_size,
      block_pooling_type=self.block_pooling_type,
      stateless=self.stateless,
      seed=self.seed,
      noise_std=self.noise_std,
      reinit=self.reinit
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    optimizer.iterations.assign(0)
    p.create_slots(optimizer, weights)

    mask_before_update = optimizer.get_slot(weight, 'mask').read_value()
    self.assertAllEqual(np.count_nonzero(mask_before_update), self.target_sparisty)

    next_step = optimizer.iterations.assign_add(1)
    reset_momentum, new_connections = p.update_masks(sparse_vars, next_step)
    self.assertAllEqual(reset_momentum, False)
    self.assertAllEqual(new_connections, None)

    mask_after_update = mask.read_value()
    self.assertAllEqual(np.count_nonzero(mask_after_update), self.target_sparsity)
    self.assertAllEqual(mask_after_update, mask_before_update) # no update

  def testGetGrowGrads(self):
    mask = tf.ones((100))
    grads = tf.

    p = pruner.RiGLPruner(
      update_schedule=self.skip_updater,
      sparsity=self.target_sparsity,
      block_size=self.block_size,
      block_pooling_type=self.block_pooling_type,
      stateless=self.stateless,
      seed=self.seed,
      noise_std=self.noise_std,
      reinit=self.reinit
    )

    grow_grads = p._get_grow_grads()
    


  def testMaskChangesAccordingtoSchedule(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100))
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        dtype=weight_dtype)
    sparse_vars = [(mask, weight, self.grad(weight))]

    p = pruner.RiGLPruner(
      update_schedule=self.skip_updater,
      sparsity=self.target_sparsity,
      block_size=self.block_size,
      block_pooling_type=self.block_pooling_type,
      stateless=self.stateless,
      seed=self.seed,
      noise_std=self.noise_std,
      reinit=self.reinit
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    optimizer.iterations.assign(0)
    p.create_slots(optimizer, weights)

    mask_before_update = optimizer.get_slot(weight, 'mask').read_value()

  # def testDropLowestMagnitudeWeights(self):

  # def testGrowHighestMagnitudeGradients(self):

  # def testRegrownConnections(self):

  # def testDropandGrowConnections(self):
  #   return


if __name__ == "__main__":
  test.main()
