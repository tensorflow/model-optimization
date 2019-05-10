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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import core_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import simple_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing import test_utils

# Abbreviated constants used in tests.
TENSORS = simple_encoder._TENSORS
PARAMS = simple_encoder._PARAMS
SHAPES = simple_encoder._SHAPES

P1_VALS = test_utils.PlusOneEncodingStage.ENCODED_VALUES_KEY
T2_VALS = test_utils.TimesTwoEncodingStage.ENCODED_VALUES_KEY
SL_VALS = test_utils.SimpleLinearEncodingStage.ENCODED_VALUES_KEY
RM_VALS = test_utils.ReduceMeanEncodingStage.ENCODED_VALUES_KEY
SIF_SIGNS = test_utils.SignIntFloatEncodingStage.ENCODED_SIGNS_KEY
SIF_INTS = test_utils.SignIntFloatEncodingStage.ENCODED_INTS_KEY
SIF_FLOATS = test_utils.SignIntFloatEncodingStage.ENCODED_FLOATS_KEY
PN_VALS = test_utils.PlusOneOverNEncodingStage.ENCODED_VALUES_KEY


class SimpleEncoderTest(tf.test.TestCase, parameterized.TestCase):

  def test_basic_encode_decode(self):
    """Tests basic encoding and decoding works as expected."""
    encoder = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(test_utils.PlusOneEncodingStage()).make())

    x = tf.constant(1.0)
    encoded_x, decode_fn = encoder.encode(x)
    decoded_x = decode_fn(encoded_x)

    x, encoded_x, decoded_x = self.evaluate([x, encoded_x, decoded_x])
    self.assertAllClose(x, decoded_x)
    self.assertAllClose(2.0, _encoded_x_field(encoded_x, [TENSORS, P1_VALS]))

  def test_encode_multiple_objects(self):
    """Tests the same object can encode multiple different objects."""
    encoder = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(test_utils.PlusOneEncodingStage()).make())

    for shape in [(2,), (2, 3), (2, 3, 4)]:
      x = tf.constant(np.ones(shape, np.float32))
      encoded_x, decode_fn = encoder.encode(x)
      decoded_x = decode_fn(encoded_x)

      x, encoded_x, decoded_x = self.evaluate([x, encoded_x, decoded_x])
      self.assertAllClose(x, decoded_x)
      self.assertAllClose(
          2.0 * np.ones(shape, np.float32),
          _encoded_x_field(encoded_x, [TENSORS, P1_VALS]))

  def test_composite_encoder(self):
    """Tests functionality with a general, composite `Encoder`."""
    encoder = core_encoder.EncoderComposer(
        test_utils.SignIntFloatEncodingStage())
    encoder.add_child(test_utils.TimesTwoEncodingStage(), SIF_SIGNS)
    encoder.add_child(test_utils.PlusOneEncodingStage(), SIF_INTS)
    encoder.add_child(test_utils.TimesTwoEncodingStage(), SIF_FLOATS).add_child(
        test_utils.PlusOneEncodingStage(), T2_VALS)
    encoder = simple_encoder.SimpleEncoder(encoder.make())

    x = tf.constant(1.2)
    encoded_x, decode_fn = encoder.encode(x)
    decoded_x = decode_fn(encoded_x)

    x_py, encoded_x_py, decoded_x_py = self.evaluate([x, encoded_x, decoded_x])
    self.assertAllClose(x_py, decoded_x_py)
    self.assertAllClose(
        2.0, _encoded_x_field(encoded_x_py, [TENSORS, SIF_SIGNS, T2_VALS]))
    self.assertAllClose(
        2.0, _encoded_x_field(encoded_x_py, [TENSORS, SIF_INTS, P1_VALS]))
    self.assertAllClose(
        1.4,
        _encoded_x_field(encoded_x_py, [TENSORS, SIF_FLOATS, T2_VALS, PN_VALS]))

  def test_python_constants_not_exposed(self):
    """Tests that only TensorFlow values are exposed to users."""
    encoder_py = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.SimpleLinearEncodingStage(2.0, 3.0)).make())
    a_var = tf.get_variable('a_var', initializer=2.0)
    b_var = tf.get_variable('b_var', initializer=3.0)
    encoder_tf = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.SimpleLinearEncodingStage(a_var, b_var)).make())

    x = tf.constant(1.0)
    encoded_x_py, decode_fn_py = encoder_py.encode(x)
    decoded_x_py = decode_fn_py(encoded_x_py)
    encoded_x_tf, decode_fn_tf = encoder_tf.encode(x)
    decoded_x_tf = decode_fn_tf(encoded_x_tf)

    # The encoded_x_tf should have two elements that encoded_x_py does not.
    # These correspond to the two variables created passed on to constructor of
    # encoder_tf, which are exposed as params. For encoder_py, these are python
    # integers, and should thus be hidden from users.
    self.assertLen(encoded_x_tf, len(encoded_x_py) + 2)

    # Make sure functionality is still the same.
    self.evaluate(tf.global_variables_initializer())
    x, decoded_x_py, decoded_x_tf = self.evaluate(
        [x, decoded_x_py, decoded_x_tf])
    self.assertAllClose(x, decoded_x_tf)
    self.assertAllClose(x, decoded_x_py)

  def test_decode_needs_input_shape_static(self):
    """Tests that mechanism for passing input shape works with static shape."""
    encoder = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.ReduceMeanEncodingStage()).make())

    x = tf.reshape(list(range(15)), [3, 5])
    encoded_x, decode_fn = encoder.encode(x)
    decoded_x = decode_fn(encoded_x)

    decoded_x = self.evaluate(decoded_x)
    self.assertAllEqual([[7.0] * 5] * 3, decoded_x)

  def test_decode_needs_input_shape_dynamic(self):
    """Tests that mechanism for passing input shape works with dynamic shape."""
    encoder = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.ReduceMeanEncodingStage()).make())

    x = test_utils.get_tensor_with_random_shape()
    encoded_x, decode_fn = encoder.encode(x)
    decoded_x = decode_fn(encoded_x)

    x, decoded_x = self.evaluate([x, decoded_x])
    self.assertAllEqual(x.shape, decoded_x.shape)

  def test_param_control_from_outside(self):
    """Tests that behavior can be controlled from outside, if needed."""
    a_var = tf.get_variable('a_var', initializer=2.0)
    b_var = tf.get_variable('b_var', initializer=3.0)
    encoder = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.SimpleLinearEncodingStage(a_var, b_var)).make())

    x = tf.constant(1.0)
    encoded_x, decode_fn = encoder.encode(x)
    decoded_x = decode_fn(encoded_x)

    self.evaluate(tf.global_variables_initializer())
    x_py, encoded_x_py, decoded_x_py = self.evaluate([x, encoded_x, decoded_x])
    self.assertAllClose(x_py, decoded_x_py)
    self.assertAllClose(5.0, _encoded_x_field(encoded_x_py, [TENSORS, SL_VALS]))

    # Change to variables should change the behavior of the encoder.
    self.evaluate([tf.assign(a_var, 5.0), tf.assign(b_var, -7.0)])
    x_py, encoded_x_py, decoded_x_py = self.evaluate([x, encoded_x, decoded_x])
    self.assertAllClose(x_py, decoded_x_py)
    self.assertAllClose(-2.0, _encoded_x_field(encoded_x_py,
                                               [TENSORS, SL_VALS]))

  @parameterized.parameters([1.0, 'str', object])
  def test_initializer_raises(self, not_an_encoder):
    """Tests invalid encoder argument."""
    with self.assertRaisesRegex(TypeError, 'Encoder'):
      simple_encoder.SimpleEncoder(not_an_encoder)

  def test_modifying_encoded_x_raises(self):
    """Tests decode_fn raises if the encoded_x dictionary is modified."""
    encoder = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(test_utils.PlusOneEncodingStage()).make())

    x = tf.constant(1.0)
    encoded_x, decode_fn = encoder.encode(x)
    encoded_x['__NOT_EXPECTED_KEY__'] = None
    with self.assertRaises(ValueError):
      decode_fn(encoded_x)
    with self.assertRaises(ValueError):
      decode_fn({})


class StatefulSimpleEncoderTest(tf.test.TestCase, parameterized.TestCase):

  def test_basic_encode_decode(self):
    """Tests basic encoding and decoding works as expected."""
    encoder = simple_encoder.StatefulSimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.PlusOneOverNEncodingStage()).make())

    x = tf.constant(1.0)
    encoder.initialize()
    encoded_x, decode_fn = encoder.encode(x)
    decoded_x = decode_fn(encoded_x)

    self.evaluate(tf.global_variables_initializer())
    for i in range(1, 10):
      x_py, encoded_x_py, decoded_x_py = self.evaluate(
          [x, encoded_x, decoded_x])
      self.assertAllClose(x_py, decoded_x_py)
      self.assertAllClose(1.0 + 1 / i,
                          _encoded_x_field(encoded_x_py, [TENSORS, PN_VALS]))

  def test_composite_encoder(self):
    """Tests functionality with a general, composite `Encoder`."""
    encoder = core_encoder.EncoderComposer(
        test_utils.SignIntFloatEncodingStage())
    encoder.add_child(test_utils.TimesTwoEncodingStage(), SIF_SIGNS)
    encoder.add_child(test_utils.PlusOneEncodingStage(), SIF_INTS)
    encoder.add_child(test_utils.TimesTwoEncodingStage(), SIF_FLOATS).add_child(
        test_utils.PlusOneOverNEncodingStage(), T2_VALS)
    encoder = simple_encoder.StatefulSimpleEncoder(encoder.make())

    x = tf.constant(1.2)
    encoder.initialize()
    encoded_x, decode_fn = encoder.encode(x)
    decoded_x = decode_fn(encoded_x)

    self.evaluate(tf.global_variables_initializer())
    for i in range(1, 10):
      x_py, encoded_x_py, decoded_x_py = self.evaluate(
          [x, encoded_x, decoded_x])
      self.assertAllClose(x_py, decoded_x_py)
      self.assertAllClose(
          2.0, _encoded_x_field(encoded_x_py, [TENSORS, SIF_SIGNS, T2_VALS]))
      self.assertAllClose(
          2.0, _encoded_x_field(encoded_x_py, [TENSORS, SIF_INTS, P1_VALS]))
      self.assertAllClose(
          0.4 + 1 / i,
          _encoded_x_field(encoded_x_py,
                           [TENSORS, SIF_FLOATS, T2_VALS, PN_VALS]))

  def test_python_constants_not_exposed(self):
    """Tests that only TensorFlow values are exposed to users."""
    encoder_py = simple_encoder.StatefulSimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.SimpleLinearEncodingStage(2.0, 3.0)).make())
    a_var = tf.get_variable('a_var', initializer=2.0)
    b_var = tf.get_variable('b_var', initializer=3.0)
    encoder_tf = simple_encoder.StatefulSimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.SimpleLinearEncodingStage(a_var, b_var)).make())

    x = tf.constant(1.0)
    encoder_py.initialize()
    encoder_tf.initialize()
    encoded_x_py, decode_fn_py = encoder_py.encode(x)
    decoded_x_py = decode_fn_py(encoded_x_py)
    encoded_x_tf, decode_fn_tf = encoder_tf.encode(x)
    decoded_x_tf = decode_fn_tf(encoded_x_tf)

    # The encoded_x_tf should have two elements that encoded_x_py does not.
    # These correspond to the two variables created passed on to constructor of
    # encoder_tf, which are exposed as params. For encoder_py, these are python
    # integers, and should thus be hidden from users.
    self.assertLen(encoded_x_tf, len(encoded_x_py) + 2)

    # Make sure functionality is still the same.
    self.evaluate(tf.global_variables_initializer())
    x, decoded_x_py, decoded_x_tf = self.evaluate(
        [x, decoded_x_py, decoded_x_tf])
    self.assertAllClose(x, decoded_x_tf)
    self.assertAllClose(x, decoded_x_py)

  def test_decode_needs_input_shape_static(self):
    """Tests that mechanism for passing input shape works with static shape."""
    encoder = simple_encoder.StatefulSimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.ReduceMeanEncodingStage()).make())

    x = tf.reshape(list(range(15)), [3, 5])
    encoder.initialize()
    encoded_x, decode_fn = encoder.encode(x)
    decoded_x = decode_fn(encoded_x)

    self.evaluate(tf.global_variables_initializer())
    decoded_x = self.evaluate(decoded_x)
    self.assertAllEqual([[7.0] * 5] * 3, decoded_x)

  def test_decode_needs_input_shape_dynamic(self):
    """Tests that mechanism for passing input shape works with dynamic shape."""
    encoder = simple_encoder.StatefulSimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.ReduceMeanEncodingStage()).make())

    x = test_utils.get_tensor_with_random_shape()
    encoder.initialize()
    encoded_x, decode_fn = encoder.encode(x)
    decoded_x = decode_fn(encoded_x)

    self.evaluate(tf.global_variables_initializer())
    x, decoded_x = self.evaluate([x, decoded_x])
    self.assertAllEqual(x.shape, decoded_x.shape)

  @parameterized.parameters([1.0, 'str', object])
  def test_initializer_raises(self, not_an_encoder):
    """Tests invalid encoder argument."""
    with self.assertRaisesRegex(TypeError, 'Encoder'):
      simple_encoder.StatefulSimpleEncoder(not_an_encoder)

  def test_multiple_initialize_raises(self):
    """Tests encoder can be initialized only once."""
    encoder = simple_encoder.StatefulSimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.PlusOneOverNEncodingStage()).make())
    encoder.initialize()
    with self.assertRaisesRegex(RuntimeError, 'already initialized'):
      encoder.initialize()

  def test_uninitialized_encode_raises(self):
    """Tests uninitialized stateful encoder cannot perform encode."""
    encoder = simple_encoder.StatefulSimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.PlusOneOverNEncodingStage()).make())
    x = tf.constant(1.0)
    with self.assertRaisesRegex(RuntimeError, 'not been initialized'):
      encoder.encode(x)

  def test_multiple_encode_raises(self):
    """Tests the encode method of stateful encoder can only be called once."""
    encoder = simple_encoder.StatefulSimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.PlusOneOverNEncodingStage()).make())
    encoder.initialize()
    x = tf.constant(1.0)
    encoder.encode(x)
    with self.assertRaisesRegex(RuntimeError, 'only once'):
      encoder.encode(x)

  def test_modifying_encoded_x_raises(self):
    """Tests decode_fn raises if the encoded_x dictionary is modified."""
    encoder = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.PlusOneOverNEncodingStage()).make())

    x = tf.constant(1.0)
    encoded_x, decode_fn = encoder.encode(x)
    encoded_x['__NOT_EXPECTED_KEY__'] = None
    with self.assertRaises(ValueError):
      decode_fn(encoded_x)
    with self.assertRaises(ValueError):
      decode_fn({})


def _encoded_x_field(encoded_x, path):
  """Returns a field from `encoded_x` returned by the `encode` method.

  In order to test the correctness of encoding, we also need to access the
  encoded objects, which in turns depends on an implementation detail (the
  specific use of `nest.flatten_with_joined_string_paths`). This dependence is
  constrained to a single place in this utility.

  Args:
    encoded_x: The structure returned by the `encode` method.
    path: A list of keys corresponding to the path in the nested dictionary
      before it was flattened.

  Returns:
    A value from `encoded_x` corresponding to the `path`.
  """
  return encoded_x['/'.join(path)]


if __name__ == '__main__':
  tf.test.main()
