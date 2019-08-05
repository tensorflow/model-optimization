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
"""Quantizer classes which implement quantization using TF Ops on a tensor.

Module: tfmot.quantization.keras
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from tensorflow_model_optimization.python.core.quantization.keras import quant_ops


@six.add_metaclass(abc.ABCMeta)
class Quantizer(object):
  """ABC interface which contains logic to quantize a tensor."""

  # TODO(pulkitb): Figure out a clean way to handle TF variables in Quantizer.
  # Currently, Quantizers need to create variables for tracking min/max values.
  # However, variables in Keras are a property of the layer and need to be
  # serialized/deserialized along with the layer.
  # For now, passing in variables as additional **kwargs in `__call__`.

  @abc.abstractmethod
  def __call__(self, inputs, step, training, **kwargs):
    """Apply quantization to the input tensor.

    The `step` variable allows a user to design a custom quantizer which
    modifies quantization behavior as training progresses.

    Args:
      inputs: Input tensor to be quantized.
      step: Current step in graph execution.
      training: Whether the graph is currently training.
      **kwargs: Additional variables which may be passed to the quantizer.

    Returns: quantized tensor.
    """


class LastValueQuantizer(Quantizer):
  """Quantize tensor based on range the last batch of values."""

  # TODO(pulkitb): Decide and change num_bits to num_fixedpoint_values.

  def __init__(self, num_bits, per_axis, symmetric):
    """Construct a LastValueQuantizer.

    Args:
      num_bits: Number of bits for quantization
      per_axis: Whether to apply per_axis quantization.
      symmetric: If true, use symmetric quantization limits instead of training
        the minimum and maximum of each quantization range separately.
    """
    self.num_bits = num_bits
    self.per_axis = per_axis
    self.symmetric = symmetric

  def __call__(self, inputs, step, training, **kwargs):
    """Quantize tensor.

    Args:
      inputs: Input tensor to be quantized.
      step: Current step in graph execution.
      training: Whether the graph is currently training.
      **kwargs: Contains `min_var` and `max_var` tf variables.

    Returns:
      Quantized tensor.
    """
    return quant_ops.LastValueQuantize(
        inputs,
        kwargs['min_var'],
        kwargs['max_var'],
        is_training=training,
        num_bits=self.num_bits,
        per_channel=self.per_axis,
        symmetric=self.symmetric,
        narrow_range=True
        # TODO(pulkitb): Figure out a clean way to use name_prefix here.
    )


class MovingAverageQuantizer(Quantizer):
  """Quantize tensor based on a moving average of values across batches."""

  def __init__(self, num_bits, per_axis, symmetric):
    """Construct a LastValueQuantizer.

    Args:
      num_bits: Number of bits for quantization
      per_axis: Whether to apply per_axis quantization.
      symmetric: If true, use symmetric quantization limits instead of training
        the minimum and maximum of each quantization range separately.
    """
    self.num_bits = num_bits
    self.per_axis = per_axis
    self.symmetric = symmetric

  def __call__(self, inputs, step, training, **kwargs):
    """Quantize tensor.

    Args:
      inputs: Input tensor to be quantized.
      step: Current step in graph execution.
      training: Whether the graph is currently training.
      **kwargs: Contains `min_var` and `max_var` tf variables.

    Returns:
      Quantized tensor.
    """
    return quant_ops.MovingAvgQuantize(
        inputs,
        kwargs['min_var'],
        kwargs['max_var'],
        ema_decay=0.999,
        is_training=training,
        num_bits=self.num_bits,
        per_channel=self.per_axis,
        symmetric=self.symmetric,
        narrow_range=False,
        # TODO(pulkitb): Figure out a clean way to use name_prefix here.
    )
