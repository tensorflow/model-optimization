# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Metrics."""

import mock
import tensorflow as tf

from tensorflow.python.eager import monitoring
from tensorflow_model_optimization.python.core.keras import metrics


class MetricsTest(tf.test.TestCase):

  gauge = monitoring.BoolGauge('/tfmot/metrics/testing', 'testing', 'labels')

  def setUp(self):
    super(MetricsTest, self).setUp()
    self.test_label = tf.keras.layers.Conv2D(1, 1).__class__.__name__
    for label in [
        self.test_label, metrics.MonitorBoolGauge._SUCCESS_LABEL,
        metrics.MonitorBoolGauge._FAILURE_LABEL
    ]:
      MetricsTest.gauge.get_cell(label).set(False)

    with mock.patch.object(metrics.MonitorBoolGauge, 'get_usage_gauge',
                           return_value=MetricsTest.gauge):
      self.monitor = metrics.MonitorBoolGauge('testing')

  def test_DecoratorTest(self):
    @self.monitor
    def func(x):
      return x + 1

    self.assertEqual(func(1), 2)
    self.assertTrue(MetricsTest.gauge.get_cell(
        metrics.MonitorBoolGauge._SUCCESS_LABEL).value())

  def test_DecoratorFailureTest(self):
    @self.monitor
    def func(x):
      raise ValueError()

    with self.assertRaises(ValueError):
      func(1)
    self.assertTrue(MetricsTest.gauge.get_cell(
        metrics.MonitorBoolGauge._FAILURE_LABEL).value())

  def test_UndecoratedTest(self):
    with self.assertRaises(ValueError):
      @metrics.MonitorBoolGauge('unknown')
      def func(x):
        return x+1
      func(1)

  def test_SetTest(self):
    self.monitor.set(self.test_label)
    self.assertTrue(MetricsTest.gauge.get_cell(self.test_label).value())


if __name__ == '__main__':
  tf.test.main()
