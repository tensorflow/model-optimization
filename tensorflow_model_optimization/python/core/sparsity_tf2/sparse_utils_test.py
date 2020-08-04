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

class SparseUtilsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(SparseUtilsTest, self).setUp()
    self.block_size = (1, 1)
    self.block_pooling_type = "AVG"
    self.target_sparsity = 0.5
    self.constant_sparsity = pruning_schedule.ConstantSparsity(self.target_sparsity, 0, 100, 1)
    self.grad = _dummy_gradient

  # setUp() lies outside of the "eager scope" that wraps the test cases
  # themselves, resulting in initializing graph tensors instead of eager
  # tensors when testing eager execution.

  def testBernouilliMaskInitializer(self):
    
