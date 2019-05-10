# Copyright 2019, The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stateless and stateful Encoder classes for encoding in "one-to-many" case."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import core_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils import py_utils


nest = tf.contrib.framework.nest

_TENSORS = 'encoded_tensors'
_PARAMS = 'params'
_SHAPES = 'shapes'


class SimpleEncoder(object):
  """A simple class for encoding.

  This class provides functionality for encoding in the "one-to-many" case,
  where a `Tensor` is encoded in one location, and is to be decoded at
  potentially many other locations, leaving the communication up to the user.

  An instance of this class can be used to encode multiple objects of different
  shape, and exposes only the `encode` method, acting as a purely functional
  transformation, i.e., not creating any TensorFlow variables.

  Note that if `encoder` contains an adaptive encoding stage, the mechanism for
  updating state will be ignored. If you intend to use the changing state during
  multiple iterations of encoding, use `StatefulSimpleEncoder` instead.
  """

  def __init__(self, encoder):
    """Constructor for `SimpleEncoder`.

    Args:
      encoder: An `Encoder` object to be used for encoding.

    Raises:
      TypeError: If `encoder` is not an instance of `Encoder`.
    """
    if not isinstance(encoder, core_encoder.Encoder):
      raise TypeError('The encoder must be an instance of `Encoder`.')
    self._encoder = encoder

  def encode(self, x, name=None):
    """Encodes a given `Tensor`.

    Args:
      x: A `Tensor`, input to be encoded.
      name: `string`, name of the operation.

    Returns:
      A tuple `(encoded_structure, decode_fn)`, where these are:
      `encoded_structure`: A dictionary representing the encoded `x`. The
        `string` keys correspond to the paths through the tree of encoding
        stages of the `Encoder`. The values are `Tensor` objects.
      `decode_fn`: A Python callable object, providing the decoding
        functionality in TensorFlow. This object expects a single argument,
        which needs to have exactly the same structure as the
        `encoded_structure`.
    """
    with tf.name_scope(name, 'simple_encoder_encode', [x]):
      encode_params, decode_params = self._encoder.get_params(
          self._encoder.initial_state())
      encoded_x, _, input_shapes = self._encoder.encode(x, encode_params)
      encoded_structure, decode_fn = _make_decode_fn(
          self._encoder, encoded_x, decode_params, input_shapes)
      return encoded_structure, decode_fn


class StatefulSimpleEncoder(object):
  """A simple stateful class for encoding.

  This class provides functionality for encoding in the "one-to-many" case,
  where a `Tensor` is encoded in one location, and is to be decoded at
  potentially many other locations, leaving the communication up to the user.

  An instance of this class maintains its own TensorFlow state. The object can
  be used to encode only a single `Tensor`, and the `initialize` method needs to
  be called before the `encode` method. The `initialize` method will create
  TensorFlow `Variable` objects representing the state of the underlying
  `Encoder`, and update them accordingly each time the encoding is executed.
  Note that the stateful mode is useful only if the encoding is to happen
  multiple times in an iterative fashion.
  """

  def __init__(self, encoder):
    """Constructor for `StatefulSimpleEncoder`.

    Args:
      encoder: An `Encoder` object to be used for encoding.

    Raises:
      TypeError: If `encoder` is not an instance of `Encoder`.
    """
    if not isinstance(encoder, core_encoder.Encoder):
      raise TypeError('The encoder must be an instance of `Encoder`.')
    self._encoder = encoder
    self._initialized = False
    self._encode_called = False

  def initialize(self):
    """Initializes the `StatefulSimpleEncoder`.

    This method creates TensorFlow variables that will be used for storing the
    state necessary for the `encoder` provided in the initializer.

    This method needs to be called before the `encode` method, and can be called
    only once.

    Raises:
      RuntimeError: If `self` was already initialized.
    """
    if self._initialized:
      raise RuntimeError('This StatefulSimpleEncoder was already initialized.')
    self._initialized = True

    def _create_state_variable_fn(path, initial_state_tensor):
      var = tf.get_variable(
          path + '_stateful_simple_encoder_state_var',
          initializer=initial_state_tensor)
      return var

    # Create the state variables and store as private attribute.
    self._state = nest.map_structure_with_paths(_create_state_variable_fn,
                                                self._encoder.initial_state())

  def encode(self, x, name=None):
    """Encodes a given `Tensor`.

    Executing the returned `encoded_x` will also update the state variables as a
    side effect after the encoding is done.

    Args:
      x: A `Tensor`, input to be encoded.
      name: `string`, name of the operation.

    Returns:
      A tuple `(encoded_x, decode_fn)`, where these are:
      `encoded_structure`: A dictionary representing the encoded `x`. The
        `string` keys correspond to the paths through the tree of encoding
        stages of the `Encoder`. The values are `Tensor` objects.
      `decode_fn`: A Python callable object, providing the decoding
        functionality in TensorFlow. This object expects a single argument,
        which needs to have exactly the same structure as the `encoded_x`
        dictionary.

    Raises:
      RuntimeError: If `self` was not initialized yet, or if the `encode` method
        was already called.
    """
    with tf.name_scope(name, 'stateful_simple_encoder_encode', [x]):
      if not self._initialized:
        raise RuntimeError(
            'The StatefulSimpleEncoder has not been initialized yet.')
      if self._encode_called:
        raise RuntimeError(
            'The encode method of the StatefulSimpleEncoder can be called only '
            'once. Because this class controls the state, this could result in '
            'a very surprising behavior and is not allowed.')
      self._encode_called = True

      state = nest.map_structure(lambda var: var.read_value(), self._state)
      encode_params, decode_params = self._encoder.get_params(state)
      encoded_x, state_update_tensors, input_shapes = self._encoder.encode(
          x, encode_params)
      updated_state = self._encoder.update_state(state, state_update_tensors)
      encoded_structure, decode_fn = _make_decode_fn(
          self._encoder, encoded_x, decode_params, input_shapes)

      with tf.control_dependencies(nest.flatten(encoded_structure)):
        # The state should be updated *after* the encoding is finished.
        update_state_ops = nest.map_structure(tf.assign, self._state,
                                              updated_state)

      with tf.control_dependencies(nest.flatten(update_state_ops)):
        # This control block does two things:
        # 1. It makes sure the update_state_ops are actually executed, as a side
        # effect of encoded_structure being evaluated.
        # 2. Wraps the encoded_structure in a tf.identity. We pass a custom name
        # argument to the op, in order to preserve the naming structure of the
        # Tensors in encoded_structure, which can be very valuable for
        # debugging.
        def rebase_name_scope(tensor):
          name = tf.contrib.framework.strip_name_scope(
              tensor.name, tf.get_default_graph().get_name_scope())
          name = name[:name.rfind(':')]
          return name + '_identity'
        name_args = nest.map_structure(rebase_name_scope, encoded_structure)
        encoded_structure = nest.map_structure(tf.identity, encoded_structure,
                                               name_args)

      return encoded_structure, decode_fn


def _make_decode_fn(encoder, encoded_x, decode_params, input_shapes):
  """Utility for creating a decoding function and its arguments.

  The inputs are potentially complex, nested structures of dictionaries. See
  documentation of the `Encoder` class for more details on the structure. In
  order to expose only a simple structure to the users, this method does
  the following:

  It creates a single dictionary out of the three input arguments, `encoded_x`,
  `decode_params`, `input_shapes`. Then, using the `nest` utility, flattens the
  dictionary. We split the flat dictionary to two parts, based on whether the
  keys map to TensorFlow objects or not. Only the part with TensorFlow objects
  is exposed to users, and is the expected input to the constructed decoding
  function.

  The decoding function merges the TensorFlow values with the non-TensorFlow
  values (never exposed to users), reconstructs the complex, nested dictionary,
  before providing the values back to the `decode` method of `encoder`.

  Args:
    encoder: An `Encoder` object that was used to generate the other arguments.
    encoded_x: The `encoded_x` value returned by `encoder.encode`.
    decode_params: The `decode_params` value returned by `encoder.get_params`.
    input_shapes: The `input_shapes` value returned by `encoder.encode`.

  Returns:
    A tuple expected as the return structure of the `encode` method.
  """
  full_encoded_structure = {
      _TENSORS: encoded_x,
      _PARAMS: decode_params,
      _SHAPES: input_shapes,
  }
  flat_encoded_structure = dict(
      nest.flatten_with_joined_string_paths(
          full_encoded_structure, separator='/'))
  flat_encoded_structure_py, flat_encoded_structure_tf = (
      py_utils.split_dict_py_tf(flat_encoded_structure))

  def decode_fn(encoded_structure):
    """Decoding function corresponding to the input arguments."""
    with tf.name_scope(None, 'simple_encoder_decode',
                       nest.flatten(encoded_structure)):
      if set(encoded_structure.keys()) != set(flat_encoded_structure_tf.keys()):
        raise ValueError(
            'The provided encoded_structure has unexpected structure. Please '
            'make sure the structure of the dictionary was not changed.')
      encoded_structure = py_utils.merge_dicts(encoded_structure,
                                               flat_encoded_structure_py)
      encoded_structure = nest.pack_sequence_as(
          full_encoded_structure, nest.flatten(encoded_structure))
      return encoder.decode(encoded_structure[_TENSORS],
                            encoded_structure[_PARAMS],
                            encoded_structure[_SHAPES])

  return flat_encoded_structure_tf, decode_fn
