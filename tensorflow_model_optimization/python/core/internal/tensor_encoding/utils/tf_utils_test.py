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
import scipy
import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils import tf_utils


class FastWalshHadamardTransformTests(tf.test.TestCase, parameterized.TestCase):
  """Tests for `fast_walsh_hadamard_transform` method."""

  @parameterized.parameters([2, 4, 8, 16])
  def test_is_rotation(self, dim):
    """Tests the transform acts as a rotation."""
    x = tf.random.normal([1, dim])
    hx = tf_utils.fast_walsh_hadamard_transform(x)
    x, hx = self.evaluate([x, hx])
    # Check that x and hx are not the same, but have equal norm.
    self.assertGreater(np.linalg.norm(x - hx), 1e-3)
    self.assertAllClose(np.linalg.norm(x), np.linalg.norm(hx))

  @parameterized.parameters([1, 2, 5, 11])
  def test_apply_twice_equals_identity(self, first_dim):
    """Tests applying the transform twice is equal to identity."""
    x = tf.random.normal([first_dim, 8])
    hx = tf_utils.fast_walsh_hadamard_transform(x)
    hhx = tf_utils.fast_walsh_hadamard_transform(hx)
    x, hhx = self.evaluate([x, hhx])
    self.assertAllEqual(x.shape, hhx.shape)
    self.assertAllClose(x, hhx)

  @parameterized.parameters([[1], [1, 4, 4], [1, 1, 1, 4]])
  def test_illegal_inputs_shape(self, *dims):
    """Tests incorrect rank of the input."""
    x = tf.random.normal(dims)
    with self.assertRaisesRegexp(ValueError,
                                 'Number of dimensions of x must be 2.'):
      tf_utils.fast_walsh_hadamard_transform(x)

  @parameterized.parameters([[1, 3], [1, 7], [1, 9], [4, 3]])
  def test_illegal_inputs_power_of_two(self, *dims):
    """Tests incorrect shape of the rank 2 input."""
    x = tf.random.normal(dims)
    with self.assertRaisesRegexp(ValueError,
                                 'The dimension of x must be a power of two.'):
      tf_utils.fast_walsh_hadamard_transform(x)

  @parameterized.parameters([2, 4, 8, 16])
  def test_output_same_as_simple_python_implementation(self, dim):
    """Tests result is identical to inefficient implementation using scipy."""
    x = tf.random.normal([3, dim])
    hx_tf = tf_utils.fast_walsh_hadamard_transform(x)
    hhx_tf = tf_utils.fast_walsh_hadamard_transform(hx_tf)
    x, hx_tf, hhx_tf = self.evaluate([x, hx_tf, hhx_tf])

    hadamard_matrix = scipy.linalg.hadamard(dim)
    hx_py = np.dot(x, hadamard_matrix) / np.sqrt(dim)
    hhx_py = np.dot(hx_py, hadamard_matrix) / np.sqrt(dim)
    self.assertAllClose(hx_py, hx_tf)
    self.assertAllClose(hhx_py, hhx_tf)


class CMWCRandomSequenceTests(tf.test.TestCase, parameterized.TestCase):
  """Tests for `_cmwc_random_sequence` method."""

  @parameterized.parameters([1, 2, 99, 100, 101, 12345])
  def test_expected_output_shape(self, num_elements):
    sequence = tf_utils._cmwc_random_sequence(num_elements,
                                              tf.constant(123, tf.int64))
    self.assertAllEqual([num_elements], sequence.shape.as_list())
    self.assertEqual(tf.float64, sequence.dtype)
    sequence = self.evaluate(sequence)
    self.assertAllGreaterEqual(sequence, 0.0)
    self.assertAllLessEqual(sequence, 1.0)

  def test_deterministic_given_seed(self):
    sequence_1 = tf_utils._cmwc_random_sequence(10, tf.constant(123, tf.int64))
    sequence_2 = tf_utils._cmwc_random_sequence(10,
                                                tf.constant(120 + 3, tf.int64))
    sequence_1, sequence_2 = self.evaluate([sequence_1, sequence_2])
    self.assertAllEqual(sequence_1, sequence_2)

  def test_differs_given_different_seed(self):
    sequence_1 = tf_utils._cmwc_random_sequence(100, tf.constant(123, tf.int64))
    sequence_2 = tf_utils._cmwc_random_sequence(100, tf.constant(
        1234, tf.int64))
    sequence_1, sequence_2 = self.evaluate([sequence_1, sequence_2])
    self.assertFalse(np.array_equal(sequence_1, sequence_2))

  def test_approximately_uniform_distribution(self):
    sequence = tf_utils._cmwc_random_sequence(100000, tf.constant(
        123, tf.int64))
    sequence = self.evaluate(sequence)
    bucket_counts, _ = np.histogram(sequence, bins=10, range=(0, 1))
    self.assertAllGreaterEqual(bucket_counts, 9750)
    self.assertAllLessEqual(bucket_counts, 10250)

  def test_tensor_num_elements_raises(self):
    with self.assertRaisesRegexp(TypeError, 'must be a Python integer'):
      tf_utils._cmwc_random_sequence(
          tf.constant(10), tf.constant(123, tf.int64))

  def test_negative_num_elements_raises(self):
    with self.assertRaisesRegexp(ValueError, 'must be positive'):
      tf_utils._cmwc_random_sequence(-10, tf.constant(123, tf.int64))

  def test_python_seed_raises(self):
    with self.assertRaisesRegexp(TypeError, 'tf.int64 Tensor'):
      tf_utils._cmwc_random_sequence(10, 123)

  def test_tf_int32_seed_raises(self):
    with self.assertRaisesRegexp(TypeError, 'tf.int64 Tensor'):
      tf_utils._cmwc_random_sequence(10, tf.constant(123, tf.int32))


class RandomSignsTests(tf.test.TestCase, parameterized.TestCase):
  """Tests for `random_signs` method."""

  @parameterized.parameters([1, 10, 101])
  def test_expected_output_values(self, num_elements):
    signs = tf_utils.random_signs(num_elements, tf.constant(123, tf.int64))
    signs = self.evaluate(signs)
    self._assert_signs(signs)

  def test_both_values_present(self):
    signs = tf_utils.random_signs(1000, tf.constant(123, tf.int64))
    signs = self.evaluate(signs)
    self._assert_signs(signs)
    self.assertGreater(sum(np.isclose(1.0, signs)), 400)
    self.assertGreater(sum(np.isclose(-1.0, signs)), 400)

  @parameterized.parameters([tf.float32, tf.float64, tf.int32, tf.int64])
  def test_expected_dtype(self, dtype):
    signs = tf_utils.random_signs(10, tf.constant(123, tf.int64), dtype)
    self.assertEqual(dtype, signs.dtype)
    signs = self.evaluate(signs)
    self._assert_signs(signs)

  def test_differs_given_different_seed(self):
    signs_1 = tf_utils.random_signs(100, tf.constant(123, tf.int64))
    signs_2 = tf_utils.random_signs(100, tf.constant(1234, tf.int64))
    signs_1, signs_2 = self.evaluate([signs_1, signs_2])
    self.assertFalse(np.array_equal(signs_1, signs_2))

  def _assert_signs(self, x):
    size = len(x)
    self.assertAllEqual([True] * size,
                        np.logical_or(np.isclose(1.0, x), np.isclose(-1.0, x)))


if __name__ == '__main__':
  tf.test.main()
