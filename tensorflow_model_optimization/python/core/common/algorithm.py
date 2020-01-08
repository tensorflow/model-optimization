# Lint as: python3
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
"""Interface for specifying model optimization algorithm."""

import abc
import attr


class OptimizationAlgorithmInterface(metaclass=abc.ABCMeta):
  """Abstract base class for model optimization algorithms.

  This interface captures any algorithm that replaces representation of some or
  all weights of a model and manipulates it during training. We refer to this
  representation as `state`.

  Users are expected to implement three methods:
  - initial_state
  - get_weight_tensors
  - update_state

  The `initial_state` method returns the initial representation which should
  replace provided weights, before any training begins. The returned `StateRepr`
  object has two fields, `trainable` and `non_trainable`. Only the values
  returned under the `trainable` field will be modified during a training step.

  The `get_weight_tensors` method implements computation necessary to map the
  `state` to the weights being optimized by this algorithm.

  The `update_state` method allows `state` to be arbitrarily modified during
  training. For some algorithms, this can be simply an identity.

  Optional `add_loss` method allows the implementer to add additional loss to
  the training objective based on `state`.

  The implementations of this interface must be purely functional. That is, no
  variable creation or execution side effects are allowed.
  """

  @abc.abstractmethod
  def initial_state(self, weights):
    """Creates initial state representation for the optimized `weights`.

    Args:
      weights: A list of weights of the layer to be optimized.

    Returns:
      A `StateRepr` object.
    """

  @abc.abstractmethod
  def get_weight_tensors(self, state):
    """Maps state representation to the weights being optimized.

    Args:
      state: A `StateRepr` object.

    Returns:
      A list of Tensors of the same shape and dtype as `weights` passed to the
      `initial_state` method.
    """

  # TODO(konkey): Come up with a better mechanism of (optionally) invoking this
  # method only infrequently, or never.
  @abc.abstractmethod
  def update_state(self, state):
    """Updates the state representation for the optimized `weights`.

    The state representation will be updated in each training step.

    Args:
      state: A `StateRepr` object.

    Returns:
      A `StateRepr` object.
    """

  # TODO(konkey): Make sure this really captures what we care about.
  def add_loss(self, state):
    """Returns additional loss to be added to the optimized model.

    Args:
      state: A `StateRepr` object.

    Returns:
      A scalar `Tensor` or `None`.
    """
    del state
    return None


@attr.s
class StateRepr(object):
  """A container class for representation of model optimization state.

  This is the object being passed between `OptimizationAlgorithmInterface` and
  an object implementing the algorithm in a specific model API, such as Keras.

  The available fields are, in order:
  - trainable
  - non_trainable

  Both fields can hold any collection of tensors compatible with `tf.nest`.

  The values in the `trainable` field are going to be modified during
  backpropagation in training. The values in the `non_trainable` field will not
  be modified, other than in the `update_state` method of
  `OptimizationAlgorithmInterface`.
  """
  trainable = attr.ib()
  non_trainable = attr.ib()
