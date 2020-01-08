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
"""Implementation of model optimization algorithm as a Keras Layer wrapper."""

import abc
import tensorflow.compat.v2 as tf

from tensorflow_model_optimization.python.core.common import algorithm

K = tf.keras.backend


class ModelOptimizationWrapper(tf.keras.layers.Layer):
  """Keras wrapper for model optimization algorithm.

  This class realizes `OptimizationAlgorithmInterface` as a wrapper of an
  existing Keras layer. The class itself is a subclass of
  `tf.keras.layers.Layer`.

  It takes an instantiated `tf.keras.layers.Layer` object, a reference to a
  subset of its weights to be optimized, and an instance of
  `OptimizationAlgorithmInterface` which will be used to optimize those weights.

  It is realized by calling the `initial_state` in the `self.build` method, and
  creating corresponding state variables for those returned tensors. The
  original weights are recovered via the `get_weight_tensors` method from those
  state variables. In the `self.call` method, the `update_state` method is
  invoked before invoking the wrapped layer's `call` method.
  """

  def __init__(self, layer, weight_attrs_to_optimize, optimization_algorithm,
               **kwargs):
    """Initializer for `ModelOptimizationWrapper`.

    TODO(konkey): Are **kwargs actually needed?

    Args:
      layer: An instance of a Keras `Layer` to be wrapped.
      weight_attrs_to_optimize: A list of weight attributes to be optimized
        using `optimization_algorithm`.
      optimization_algorithm: An `OptimizationAlgorithmInterface`.
      **kwargs: Additional keyword args to be passed to the base class.

    Raises:
      RuntimeError: If not executing in graph mode.
    """
    if tf.executing_eagerly():  # TODO(konkey): Fix eager execution.
      raise RuntimeError('WIP! Please use graph mode for now.')
    assert not isinstance(layer, ModelOptimizationWrapper)
    self.layer = layer
    self._weight_attrs_to_optimize = weight_attrs_to_optimize
    assert isinstance(optimization_algorithm,
                      algorithm.OptimizationAlgorithmInterface)
    self._algorithm = optimization_algorithm
    super(ModelOptimizationWrapper, self).__init__(**kwargs)

  def build(self, input_shape):
    if not self.layer.built:
      self.layer.build(input_shape)

    weights = self._get_weights_to_optimize()
    self._original_weights = weights  # PROTOTYPE HACK.
    self._state_vars = self._add_state_variables(
        self._algorithm.initial_state(weights))
    self._add_algorithm_loss(self._state_vars)

    self.built = True

  def call(self, inputs, training=None):
    if training is None:
      training = tf.convert_to_tensor(K.learning_phase())

    if tf.executing_eagerly():  # TODO(konkey): Fix eager execution.
      raise RuntimeError('WIP! Please use graph mode for now.')
    else:
      return tf.cond(training, lambda: self._graph_training_call(inputs),
                     lambda: self.layer.call(inputs))

  def _graph_training_call(self, inputs):
    # TODO(konkey): Understand the racing condition without .read_value().
    updated_state = self._algorithm.update_state(
        tf.nest.map_structure(lambda v: v.read_value(), self._state_vars))
    control_deps = tf.nest.flatten(
        tf.nest.map_structure(lambda var, value: var.assign(value),
                              self._state_vars, updated_state))

    with tf.control_dependencies(control_deps):
      # The weight tensors will always need to be re-computed before being used
      # in the underlying layer's `call` method.
      self._set_weights_to_optimize(self._state_vars)
    return self.layer.call(inputs)

  def _get_weights_to_optimize(self):
    return [
        getattr(self.layer, w_attr) for w_attr in self._weight_attrs_to_optimize
    ]

  def _set_weights_to_optimize(self, state):
    weight_tensors = self._algorithm.get_weight_tensors(state)
    weight_tensors = tf.nest.map_structure(  # PROTOTYPE HACK.
        lambda x, y: x + 0.0 * y, weight_tensors, self._original_weights)
    for weight, w_attr in zip(weight_tensors, self._weight_attrs_to_optimize):
      setattr(self.layer, w_attr, weight)  # TODO(konkey): This is brittle.

  def _add_state_variables(self, state):
    return algorithm.StateRepr(
        tf.nest.map_structure(lambda t: self._add_custom_variables(t, True),
                              state.trainable),
        tf.nest.map_structure(lambda t: self._add_custom_variables(t, False),
                              state.non_trainable))

  def _add_custom_variables(self, tensor, trainable):
    return self.add_weight(
        shape=tensor.shape,
        initializer=tf.keras.initializers.Constant(K.get_value(tensor)),
        dtype=tensor.dtype,
        trainable=trainable)

  def _add_algorithm_loss(self, state):
    loss = self._algorithm.add_loss(state)
    if loss is not None:
      self.add_loss(loss)


class KerasOptimizationAlgorithmConfig(metaclass=abc.ABCMeta):
  """A configuration object for applying `OptimizationAlgorithmInterface`.

  This class is expected to be implemented together with an
  `OptimizationAlgorithmInterface`, describing how that algorithm is to be
  mapped to Keras-specific objects.

  Typically, an implementation would come with reasonable defaults for some of
  the common symbols in `tf.keras.layers` namespace. However, this class can be
  also implemented by an end user, wishing to apply a model optimization
  algorithm to a custom layer.
  """

  @abc.abstractmethod
  def weight_attrs_to_optimize(self, layer_cls):
    """Given a Keras Layer class, returns the weight attributes to be optimized.

    Args:
      layer_cls: A subclass of `tf.keras.layers.Layer` to be optimized.

    Returns:
      A list of strings. The list is empty if no weights are to be optimized.
    """


def wrap_keras_layer(layer, optimization_algorithm, optimization_config):
  """Wraps a Keras layer and applies `optimization_algorithm`.

  Args:
    layer: A Keras Layer to optimize.
    optimization_algorithm: A `OptimizationAlgorithmInterface` to use for
      optimization of the weights of `layer`.
    optimization_config: A `KerasOptimizationAlgorithmConfig` describing weight
      attributes to be optimized.

  Returns:
    A `ModelOptimizationWrapper`, wrapping `layer` and applying
    `optimization_algorithm`. If no weights are to be optimized, returns
    unmodified `layer`.
  """
  assert isinstance(layer, tf.keras.layers.Layer)
  assert isinstance(optimization_algorithm,
                    algorithm.OptimizationAlgorithmInterface)
  assert isinstance(optimization_config, KerasOptimizationAlgorithmConfig)

  weight_attrs_to_optimize = optimization_config.weight_attrs_to_optimize(
      layer.__class__)
  if weight_attrs_to_optimize:
    return ModelOptimizationWrapper(layer, weight_attrs_to_optimize,
                                    optimization_algorithm)
  else:
    # If no weights are to be optimized, just return the unmodified layer.
    return layer
