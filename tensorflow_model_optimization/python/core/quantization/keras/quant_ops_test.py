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
"""Tests for Quantize Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow as tf

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow_model_optimization.python.core.quantization.keras import quant_ops

_SYMMETRIC_RANGE_RATIO = 0.9921875  # 127 / 128


@keras_parameterized.run_all_keras_modes
class QuantOpsTest(tf.test.TestCase, parameterized.TestCase):

  def testLastValueQuantizeTrainingAssign(self):
    min_value, max_value = self._GetMinMaxValues(quant_ops.LastValueQuantize,
                                                 [[-1, 1]])
    self.assertEqual(min_value, -1.0)
    self.assertEqual(max_value, 1.0)

  def testLastValueSymmetricQuantizeTrainingAssign(self):
    min_value, max_value = self._GetMinMaxValues(
        quant_ops.LastValueQuantize,
        [[-_SYMMETRIC_RANGE_RATIO, _SYMMETRIC_RANGE_RATIO]],
        symmetric=True,
        narrow_range=False)
    self.assertEqual(min_value, -1.0)
    self.assertEqual(max_value, _SYMMETRIC_RANGE_RATIO)

  def testLastValueSymmetricQuantizeNarrowRangeTrainingAssign(self):
    min_value, max_value = self._GetMinMaxValues(
        quant_ops.LastValueQuantize, [[-1, 0.5]],
        symmetric=True,
        narrow_range=True)
    self.assertEqual(min_value, -1.0)
    self.assertEqual(max_value, 1)

  def testMovingAvgQuantizeTrainingAssign(self):
    min_value, max_value = self._GetMinMaxValues(quant_ops.MovingAvgQuantize,
                                                 [[-1, 1], [0, 0]])
    self.assertAlmostEqual(min_value, -0.000999, delta=1e-6)
    self.assertAlmostEqual(max_value, 0.000999, delta=1e-6)

  def testMovingAvgSymmetricQuantizeTrainingAssign(self):
    min_value, max_value = self._GetMinMaxValues(
        quant_ops.MovingAvgQuantize, [[-1, 0.5], [0, 0]], symmetric=True)
    self.assertAlmostEqual(min_value, -0.000999, delta=1e-6)
    self.assertAlmostEqual(
        max_value, 0.000999 * _SYMMETRIC_RANGE_RATIO, delta=1e-6)
    self.assertAlmostEqual(max_value, min_value * -_SYMMETRIC_RANGE_RATIO)

  def testMovingAvgSymmetricQuantizeNarrowRangeTrainingAssign(self):
    min_value, max_value = self._GetMinMaxValues(
        quant_ops.MovingAvgQuantize, [[-1, 0.5], [0, 0]],
        symmetric=True,
        narrow_range=True)
    self.assertAlmostEqual(min_value, -0.000999, delta=1e-6)
    self.assertAlmostEqual(max_value, 0.000999, delta=1e-6)
    self.assertAlmostEqual(max_value, -min_value)

  def testVariablesNotPartitioned_LastValue(self):
    # Variables added should not use a default partiioner since they are
    # scalar. There would be a tensorflow error thrown if the partitioner was
    # respected by the rewrite.
    with ops.Graph().as_default():
      with variable_scope.variable_scope(
          'part', partitioner=partitioned_variables.fixed_size_partitioner(2)):
        x = array_ops.placeholder(dtypes.float32, shape=[2])
        min_var = variables.Variable(0.0)
        max_var = variables.Variable(0.0)
        _ = quant_ops.LastValueQuantize(x, min_var, max_var, is_training=True)

  def testVariablesNotPartitioned_MovingAvg(self):
    # Variables added should not use a default partiioner since they are
    # scalar. There would be a tensorflow error thrown if the partitioner was
    # respected by the rewrite.
    with ops.Graph().as_default():
      with variable_scope.variable_scope(
          'part', partitioner=partitioned_variables.fixed_size_partitioner(2)):
        x = array_ops.placeholder(dtypes.float32, shape=[2])
        min_var = variables.Variable(0.0)
        max_var = variables.Variable(0.0)
        _ = quant_ops.MovingAvgQuantize(x, min_var, max_var, is_training=True)

  def _GetMinMaxValues(self, quantize_fn, input_values, **kwds):
    g = ops.Graph()
    with session.Session(graph=g) as sess:
      x = array_ops.placeholder(dtypes.float32, shape=[2])
      min_var = variables.Variable(0.0)
      max_var = variables.Variable(0.0)
      y = quantize_fn(x, min_var, max_var, is_training=True, **kwds)

      # Run the step.
      sess.run(variables.global_variables_initializer())
      for input_elem in input_values:
        sess.run(y, feed_dict={x: input_elem})

      # Now check that the min_max_vars were, in fact, updated.
      min_max_values = sess.run([min_var, max_var])
      return min_max_values[0], min_max_values[1]


if __name__ == '__main__':
  tf.test.main()
