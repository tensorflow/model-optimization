# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for schedulers."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.common.keras.compression import schedules


class SimpleScheduler(schedules.Scheduler):

  def __call__(self, step: int) -> float:
    return 0.1 if step >= 1000 else 0.6


class SimpleSchedulerTest(tf.test.TestCase):

  def testSimpleScheduler(self):
    scheduler = SimpleScheduler()
    expected = [0.6, 0.6, 0.1, 0.1]
    output = [scheduler(i) for i in [0, 100, 1000, 2000]]
    self.assertAllEqual(output, expected)


class CubicPolynomialDecayTest(tf.test.TestCase):

  def testBeforeDecaying(self):
    init_value = 0.1
    final_value = 1.0
    begin_step = 10
    decaying_step = 10
    total_training_step = begin_step
    scheduler = schedules.PolynomialDecay(init_value, decaying_step,
                                          final_value, begin_step=begin_step,
                                          exponent=3)
    output = [scheduler(i) for i in range(total_training_step)]
    expected = [init_value] * begin_step
    self.assertAllClose(output, expected)

  def testDecaying(self):
    init_value = 0.1
    final_value = 1.0
    begin_step = 10
    decaying_step = 10
    exponent = 3
    scheduler = schedules.PolynomialDecay(init_value, decaying_step,
                                          final_value, begin_step=begin_step,
                                          exponent=exponent)
    expected = [final_value + (init_value - final_value) * \
                (1-float(i)/decaying_step) ** exponent
                for i in range(decaying_step)]
    output = [scheduler(begin_step + i) for i in range(decaying_step)]
    self.assertAllClose(output, expected)

  def testBeyondEnd(self):
    init_value = 0.1
    final_value = 1.0
    begin_step = 10
    decaying_step = 10
    total_steps = 30
    beyond_end_steps = total_steps - decaying_step - begin_step
    scheduler = schedules.PolynomialDecay(init_value, decaying_step,
                                          final_value, begin_step=begin_step,
                                          exponent=3)
    expected = [final_value] * beyond_end_steps
    output = [scheduler(begin_step + decaying_step + i)
              for i in range(beyond_end_steps)]
    self.assertAllClose(output, expected)


class LinearPolynomialDecayTest(tf.test.TestCase):

  def testHalfWay(self):
    step = 5
    lr = 0.05
    end_lr = 0.0
    decayed_lr = schedules.PolynomialDecay(lr, 10, end_lr)
    expected = lr * 0.5
    self.assertAllClose(decayed_lr(step), expected, 1e-6)

  def testEnd(self):
    step = 10
    lr = 0.05
    end_lr = 0.001
    decayed_lr = schedules.PolynomialDecay(lr, 10, end_lr)
    expected = end_lr
    self.assertAllClose(decayed_lr(step), expected, 1e-6)

  def testHalfWayWithEnd(self):
    step = 5
    lr = 0.05
    end_lr = 0.001
    decayed_lr = schedules.PolynomialDecay(lr, 10, end_lr)
    expected = (lr + end_lr) * 0.5
    self.assertAllClose(decayed_lr(step), expected, 1e-6)

  def testBeyondEnd(self):
    step = 15
    lr = 0.05
    end_lr = 0.001
    decayed_lr = schedules.PolynomialDecay(lr, 10, end_lr)
    expected = end_lr
    self.assertAllClose(decayed_lr(step), expected, 1e-6)

if __name__ == '__main__':
  tf.test.main()
