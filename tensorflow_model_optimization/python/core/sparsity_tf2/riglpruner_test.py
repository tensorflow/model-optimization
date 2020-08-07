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
from tensorflow_model_optimization.python.core.sparsity.keras import schedule
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
    self.initial_drop_fraction = 0.3
    self.constant_update = pruning_schedule.ConstantSparsity(self.initial_drop_fraction, 0, 100, 1)
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
      update_schedule=self.constant_update,
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



    # TODO: rebase this branch off the schedule, loop prior to
    # updates and check that it has not updated
    return

  def testSameNumberofParamsEachLayer(self):

  def testDropLowestMagnitudeWeights(self):

  def testGrowHighestMagnitudeGradients(self):

  def testRegrownConnections(self):

  def testDropandGrowConnections(self):
    return


if __name__ == "__main__":
  test.main()
