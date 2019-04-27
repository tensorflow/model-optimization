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
"""Tests for Pruning Schedule."""

from absl.testing import parameterized

from tensorflow.python import keras
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule


@parameterized.parameters('constant_sparsity', 'polynomial_decay')
class PruningScheduleTest(test.TestCase, parameterized.TestCase):
  """Test to verify PruningSchedule behavior for step parameters.

  This is a parameterized test which runs over all PruningSchedule classes
  to ensure they validate parameters such as begin_step, end_step and frequency
  properly and also apply them correctly during execution.
  """

  # Argument Validation tests

  @staticmethod
  def _construct_pruning_schedule(
      schedule_type, begin_step, end_step, frequency=10):
    # Uses default values for sparsity. We're only testing begin_step, end_step
    # and frequency here.
    if schedule_type == 'constant_sparsity':
      return pruning_schedule.ConstantSparsity(
          0.5, begin_step, end_step, frequency)
    elif schedule_type == 'polynomial_decay':
      return pruning_schedule.PolynomialDecay(
          0.2, 0.8, begin_step, end_step, 3, frequency)

  def testBeginStepGreaterThanEqualsZero(self, schedule_type):
    with self.assertRaises(ValueError):
      self._construct_pruning_schedule(schedule_type, -1, 1000)
    with self.assertRaises(ValueError):
      self._construct_pruning_schedule(schedule_type, -5, 1000)

    self._construct_pruning_schedule(schedule_type, 0, 1000)
    self._construct_pruning_schedule(schedule_type, 1, 1000)
    self._construct_pruning_schedule(schedule_type, 100, 1000)

  def testEndStepGreaterThanEqualsZero(self, schedule_type):
    with self.assertRaises(ValueError):
      self._construct_pruning_schedule(schedule_type, 10, -5)

    self._construct_pruning_schedule(schedule_type, 0, 0)
    self._construct_pruning_schedule(schedule_type, 0, 1)
    self._construct_pruning_schedule(schedule_type, 0, 100)

  def testEndStepGreaterThanEqualsBeginStep(self, schedule_type):
    with self.assertRaises(ValueError):
      self._construct_pruning_schedule(schedule_type, 10, 5)

    self._construct_pruning_schedule(schedule_type, 10, 10)
    self._construct_pruning_schedule(schedule_type, 10, 20)

  def testFrequencyIsPositive(self, schedule_type):
    with self.assertRaises(ValueError):
      self._construct_pruning_schedule(schedule_type, 10, 1000, 0)
    with self.assertRaises(ValueError):
      self._construct_pruning_schedule(schedule_type, 10, 1000, -1)
    with self.assertRaises(ValueError):
      self._construct_pruning_schedule(schedule_type, 10, 1000, -5)

    self._construct_pruning_schedule(schedule_type, 10, 1000, 1)
    self._construct_pruning_schedule(schedule_type, 10, 1000, 10)

  def _validate_sparsity(self, schedule_construct_fn):
    # Should not be < 0.0
    with self.assertRaises(ValueError):
      schedule_construct_fn(-0.001)
    with self.assertRaises(ValueError):
      schedule_construct_fn(-1.0)
    with self.assertRaises(ValueError):
      schedule_construct_fn(-10.0)

    # Should not be >= 1.0
    with self.assertRaises(ValueError):
      schedule_construct_fn(1.0)
    with self.assertRaises(ValueError):
      schedule_construct_fn(10.0)

    schedule_construct_fn(0.0)
    schedule_construct_fn(0.001)
    schedule_construct_fn(0.5)
    schedule_construct_fn(0.99)

  def testSparsityValueIsValid(self, schedule_type):
    if schedule_type == 'constant_sparsity':
      # pylint: disable=unnecessary-lambda
      self._validate_sparsity(lambda s: pruning_schedule.ConstantSparsity(s, 0))
    elif schedule_type == 'polynomial_decay':
      self._validate_sparsity(
          lambda s: pruning_schedule.PolynomialDecay(s, 0.8, 0, 10))
      self._validate_sparsity(
          lambda s: pruning_schedule.PolynomialDecay(0.2, s, 0, 10))

  # Tests to ensure begin_step, end_step, frequency are used correctly.

  @tf_test_util.run_in_graph_and_eager_modes
  def testPrunesOnlyInBeginEndStepRange(self, schedule_type):
    sparsity = self._construct_pruning_schedule(schedule_type, 100, 200, 1)

    # Before begin step
    step_90 = variables.Variable(90)
    step_99 = variables.Variable(99)
    # In range
    step_100 = variables.Variable(100)
    step_110 = variables.Variable(110)
    step_200 = variables.Variable(200)
    # After end step
    step_201 = variables.Variable(201)
    step_210 = variables.Variable(210)
    self.evaluate(variables.global_variables_initializer())

    self.assertFalse(self.evaluate(sparsity(step_90))[0])
    self.assertFalse(self.evaluate(sparsity(step_99))[0])

    self.assertTrue(self.evaluate(sparsity(step_100))[0])
    self.assertTrue(self.evaluate(sparsity(step_110))[0])
    self.assertTrue(self.evaluate(sparsity(step_200))[0])

    self.assertFalse(self.evaluate(sparsity(step_201))[0])
    self.assertFalse(self.evaluate(sparsity(step_210))[0])

  @tf_test_util.run_in_graph_and_eager_modes
  def testOnlyPrunesAtValidFrequencySteps(self, schedule_type):
    sparsity = self._construct_pruning_schedule(schedule_type, 100, 200, 10)

    step_100 = variables.Variable(100)
    step_109 = variables.Variable(109)
    step_110 = variables.Variable(110)
    step_111 = variables.Variable(111)
    self.evaluate(variables.global_variables_initializer())

    self.assertFalse(self.evaluate(sparsity(step_109))[0])
    self.assertFalse(self.evaluate(sparsity(step_111))[0])

    self.assertTrue(self.evaluate(sparsity(step_100))[0])
    self.assertTrue(self.evaluate(sparsity(step_110))[0])


class ConstantSparsityTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes
  def testPrunesForeverIfEndStepIsNegativeOne(self):
    sparsity = pruning_schedule.ConstantSparsity(0.5, 0, -1, 10)

    step_10000 = variables.Variable(10000)
    step_100000000 = variables.Variable(100000000)
    self.evaluate(variables.global_variables_initializer())

    self.assertTrue(self.evaluate(sparsity(step_10000))[0])
    self.assertTrue(self.evaluate(sparsity(step_100000000))[0])

    self.assertAllClose(0.5, self.evaluate(sparsity(step_10000))[1])
    self.assertAllClose(0.5, self.evaluate(sparsity(step_100000000))[1])

  @tf_test_util.run_in_graph_and_eager_modes
  def testPrunesWithConstantSparsity(self):
    sparsity = pruning_schedule.ConstantSparsity(0.5, 100, 200, 10)

    step_100 = variables.Variable(100)
    step_110 = variables.Variable(110)
    step_200 = variables.Variable(200)
    self.evaluate(variables.global_variables_initializer())

    self.assertAllClose(0.5, self.evaluate(sparsity(step_100))[1])
    self.assertAllClose(0.5, self.evaluate(sparsity(step_110))[1])
    self.assertAllClose(0.5, self.evaluate(sparsity(step_200))[1])

  def testSerializeDeserialize(self):
    sparsity = pruning_schedule.ConstantSparsity(0.7, 10, 20, 10)

    config = sparsity.get_config()
    sparsity_deserialized = keras.utils.generic_utils.deserialize_keras_object(
        config, custom_objects={
            'ConstantSparsity': pruning_schedule.ConstantSparsity,
            'PolynomialDecay': pruning_schedule.PolynomialDecay
        })

    self.assertEqual(sparsity.__dict__, sparsity_deserialized.__dict__)


class PolynomialDecayTest(test.TestCase):

  def testRaisesErrorIfEndStepIsNegative(self):
    with self.assertRaises(ValueError):
      pruning_schedule.PolynomialDecay(0.4, 0.8, 10, -1)

  @tf_test_util.run_in_graph_and_eager_modes
  def testPolynomialDecay_PrunesCorrectly(self):
    sparsity = pruning_schedule.PolynomialDecay(0.2, 0.8, 100, 110, 3, 2)

    step_100 = variables.Variable(100)
    step_102 = variables.Variable(102)
    step_105 = variables.Variable(105)
    step_110 = variables.Variable(110)
    self.evaluate(variables.global_variables_initializer())

    # These values were generated using tf.polynomial_decay with the same
    # params in a colab to verify.
    self.assertAllClose(0.2, self.evaluate(sparsity(step_100))[1])
    self.assertAllClose(0.4928, self.evaluate(sparsity(step_102))[1])
    self.assertAllClose(0.725, self.evaluate(sparsity(step_105))[1])
    self.assertAllClose(0.8, self.evaluate(sparsity(step_110))[1])

  def testSerializeDeserialize(self):
    sparsity = pruning_schedule.PolynomialDecay(0.2, 0.6, 10, 20, 5, 10)

    config = sparsity.get_config()
    sparsity_deserialized = keras.utils.generic_utils.deserialize_keras_object(
        config, custom_objects={
            'ConstantSparsity': pruning_schedule.ConstantSparsity,
            'PolynomialDecay': pruning_schedule.PolynomialDecay
        })

    self.assertEqual(sparsity.__dict__, sparsity_deserialized.__dict__)


if __name__ == '__main__':
  test.main()
