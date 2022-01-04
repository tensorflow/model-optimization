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
"""Implements monitoring."""

import functools

from tensorflow.python.eager import monitoring


class MonitorBoolGauge():
  """Monitoring utility class for usage metrics."""

  _PRUNE_FOR_BENCHMARK_USAGE = monitoring.BoolGauge(
      '/tfmot/api/sparsity/prune_for_benchmark',
      'prune_for_benchmark usage.', 'status')

  _PRUNE_LOW_MAGNITUDE_USAGE = monitoring.BoolGauge(
      '/tfmot/api/sparsity/prune_low_magnitude',
      'prune_low_magnitude usage.', 'status')

  _PRUNE_WRAPPER_USAGE = monitoring.BoolGauge(
      '/tfmot/api/sparsity/pruning_wrapper',
      'Pruning wrapper class usage.', 'layer')

  _QUANTIZE_APPLY_USAGE = monitoring.BoolGauge(
      '/tfmot/api/quantization/quantize_apply',
      'quantize_apply usage.', 'status')

  _QUANTIZE_WRAPPER_USAGE = monitoring.BoolGauge(
      '/tfmot/api/quantization/quantize_wrapper',
      'Quantize wrapper class usage.', 'layer')

  _SUCCESS_LABEL = 'success'
  _FAILURE_LABEL = 'failure'

  def __init__(self, name):
    self.bool_gauge = self.get_usage_gauge(name)

  def get_usage_gauge(self, name):
    """Gets a gauge by name."""
    if name == 'prune_for_benchmark_usage':
      return MonitorBoolGauge._PRUNE_FOR_BENCHMARK_USAGE
    if name == 'prune_low_magnitude_usage':
      return MonitorBoolGauge._PRUNE_LOW_MAGNITUDE_USAGE
    if name == 'prune_low_magnitude_wrapper_usage':
      return MonitorBoolGauge._PRUNE_WRAPPER_USAGE
    if name == 'quantize_apply_usage':
      return MonitorBoolGauge._QUANTIZE_APPLY_USAGE
    if name == 'quantize_wrapper_usage':
      return MonitorBoolGauge._QUANTIZE_WRAPPER_USAGE
    raise ValueError('Invalid gauge name: {}'.format(name))

  def __call__(self, func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
      try:
        results = func(*args, **kwargs)
        self.bool_gauge.get_cell(MonitorBoolGauge._SUCCESS_LABEL).set(True)
        return results
      except Exception as error:
        self.bool_gauge.get_cell(MonitorBoolGauge._FAILURE_LABEL).set(True)
        raise error

    if self.bool_gauge:
      return inner

    return func

  def set(self, label=None, value=True):
    """Set the bool gauge to value if initialized.

    Args:
      label: optional string label defaults to None.
      value: optional bool value defaults to True.
    """
    if self.bool_gauge:
      self.bool_gauge.get_cell(label).set(value)
