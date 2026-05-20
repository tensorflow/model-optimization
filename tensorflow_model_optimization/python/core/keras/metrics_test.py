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
import sys
import tensorflow as tf

from tensorflow.python.eager import monitoring
from tensorflow_model_optimization.python.core.keras import metrics
from tensorflow_model_optimization.python.core.keras import compat
from tensorflow_model_optimization.python.core.keras.compat import keras


class MetricsTest(tf.test.TestCase):

  gauge = monitoring.BoolGauge('/tfmot/metrics/testing', 'testing', 'labels')

  def setUp(self):
    super(MetricsTest, self).setUp()
    self.test_label = keras.layers.Conv2D(1, 1).__class__.__name__
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


class CompatGetKerasInstanceTest(tf.test.TestCase):
  """Tests for _get_keras_instance fallback when tf_keras is unavailable."""

  def test_fallback_to_tf_keras_when_tf_keras_not_installed(self):
    """When tf.keras reports v3 but tf_keras is missing, fall back to tf.keras."""
    original_modules = dict(sys.modules)
    # Remove tf_keras from sys.modules so the import inside the function fails.
    sys.modules.pop('tf_keras', None)
    try:
      with mock.patch.object(tf.keras, 'version', return_value='3.0.0',
                             create=True):
        with mock.patch.dict(sys.modules, {'tf_keras': None}):
          result = compat._get_keras_instance()
          self.assertIs(result, tf.keras)
    finally:
      # Restore original sys.modules state.
      for key in list(sys.modules):
        if key not in original_modules:
          del sys.modules[key]


if __name__ == '__main__':
  tf.test.main()
