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
"""Compression Scheduler for tfmot compression."""
import abc
from typing import Union, Optional

import tensorflow as tf


class Scheduler(metaclass=abc.ABCMeta):
  """Abstract Scheduler."""

  @abc.abstractmethod
  def __call__(self, step: Union[int, tf.Tensor]) -> tf.Tensor:
    """Scheduler function given tf.Tensor step number.

    Args:
      step: tf.Tensor with tf.int32 or tf.int64 representing the current step
        number of training loops.

    Returns:
      Any tf.Tensor Scheduled value of given `step`
    """
    raise NotImplementedError()


class PolynomialDecay(Scheduler):
  """Scheduling based on polynomial equation.

  s(t) = start_value for t < begin_step

       = end_value + [(start_value - end_value) * (1 - decay_term) ** exponent]

       where decay_term = (t - begin_step) / decay_steps

             for 0 <= 1 - decay_term <= 1
                 <-> 0 <= decay_term <= 1
                 <-> 0 <= (t - begin_step) / decay_steps <= 1
                 <-> 0 <= (t - begin_step) <= decay_steps
                 <-> begin_step <= t <= begin_step + decay_steps (=end_step)

       = end_value   for t > begin_step + decay_steps (=end_step)
  """

  def __init__(self,
               start_value: Union[int, float],
               decay_steps: int,
               end_value: Union[int, float],
               begin_step: Optional[int] = 0,
               exponent: Optional[float] = 1.0,
               dtype: Optional[tf.dtypes.DType] = tf.float32,
               name: Optional[str] = None):
    """Initialize PolynomialDecayScheduler.

    Args:
      start_value: the initial value of decaying. It is also the default value
        of this scheduler for step <= begin_step.
      decay_steps: A Python positive int value for duration of decaying.
      end_value: the final value of decaying. It is also the default value of
        this scheduler for step >= end_step = begin_step + decay_steps
      begin_step: The step value that this scheduler starts decaying.
        Defaults to 0, which means it decays right after training starts.
      exponent: The exponent of the polynomial decaying.
        Defaults to 1.0, a linear function.
      dtype: `tf.dtypes.DType`, dtype of returned tensor.
        Defaults to tf.float32.
      name: A Python `str` for the name scope of this scheduler.

    Returns:
      A `tf.Tensor` of the scheduled output value calculated from the polynomial
      equation as given above.
    """
    self.name = name
    self.start_value = start_value
    self.begin_step = begin_step
    self.end_value = end_value
    self.decay_steps = decay_steps
    self.end_step = self.begin_step + self.decay_steps
    self.exponent = exponent
    self.dtype = dtype

  def __call__(self, step: Union[int, tf.Tensor]) -> tf.Tensor:

    with tf.name_scope(self.name or "PolynomialDecay"):
      val = tf.cond(tf.math.less(step, self.begin_step),
                    lambda: tf.cast(self.start_value, dtype=self.dtype),
                    lambda: self._after_begin_step(step), name="start")
    return val

  def _after_begin_step(self, step: Union[int, tf.Tensor]) -> tf.Tensor:

    with tf.name_scope(self.name or "PolynomialDecay"):
      val = tf.cond(tf.math.greater(step, self.end_step),
                    lambda: tf.cast(self.end_value, dtype=self.dtype),
                    lambda: self._during_decay(step), name="end")
    return val

  def _during_decay(self, step: Union[int, tf.Tensor]) -> tf.Tensor:
    """Return decayed scheduled value."""

    with tf.name_scope(self.name or "PolynomialDecay"):
      local_steps = tf.cast(step - self.begin_step, dtype=tf.float32)
      decay_term = tf.math.divide(local_steps,
                                  tf.cast(self.decay_steps, dtype=tf.float32))
      total_delta = tf.cast(self.start_value - self.end_value, dtype=tf.float32)
      target = tf.math.add(self.end_value, tf.cast(
          tf.math.multiply(total_delta, tf.pow(1 - decay_term, self.exponent)),
          dtype=self.dtype))
      val = tf.stop_gradient(target)
    return val
