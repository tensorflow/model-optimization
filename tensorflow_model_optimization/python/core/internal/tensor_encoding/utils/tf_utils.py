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
"""TensorFlow utilities for the `tensor_encoding` package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

import numpy as np
import tensorflow as tf


def fast_walsh_hadamard_transform(x):
  """Applies the fast Walsh-Hadamard transform to a set of vectors.

  This method uses a composition of existing TensorFlow operations to implement
  the transform.

  Args:
    x: A `Tensor`. Must be of shape `[a, b]`, where `a` can be anything (not
      necessarily known), and `b` must be a power of two, not required to be
      statically known.

  Returns:
    A `Tensor` of shape `[a, b]`, where `[i, :]` is the product `x[i, :]*H`,
      where `H` is the Hadamard matrix.

  Raises:
    ValueError: If the input is not rank 2 `Tensor`, and if the second dimension
      is statically known and is not a power of two.
    OpError: If the second dimension is not statically known and is not a power
      of two. Note that in graph execution, this error is not raised during the
      execution of the Python function, but during execution of the resulting
      computation.
  """
  with tf.compat.v1.name_scope(None, 'fast_walsh_hadamard_transform'):
    # Validate input.
    x = tf.convert_to_tensor(x)
    if x.shape.ndims != 2:
      raise ValueError(
          'Number of dimensions of x must be 2. Shape of x: %s' % x.shape)

    original_x_shape = x.shape.as_list()
    dim = x.shape.as_list()[-1]

    if dim is None:  # dim is not statically known.
      dim = tf.shape(x)[-1]
      log2 = tf.cast(
          tf.math.round(
              tf.math.log(tf.cast(dim, tf.float32)) / tf.math.log(2.)),
          tf.int32)
      with tf.control_dependencies([
          tf.compat.v1.assert_equal(
              dim,
              tf.math.pow(2, log2),
              message='The dimension of x must be a power of two.'
              'Provided dimension is: %s' % dim)
      ]):
        x = tf.identity(x)
    else:  # dim is statically known.
      if not (dim and ((dim & (dim - 1)) == 0)):
        raise ValueError('The dimension of x must be a power of two. '
                         'Provided dimension is: %s' % dim)
      log2 = int(np.ceil(np.log2(dim)))
      if dim == 1:  # Equivalent to identity.
        return tf.identity(x)

    h_core = tf.constant([[1., 1.], [1., -1.]],
                         dtype=x.dtype,
                         name='hadamard_weights_2x2')
    permutation = tf.constant([0, 2, 1], name='hadamard_permutation')

    # A step of the fast Walsh-Hadamard algorithm.
    def _hadamard_step(x, dim):
      """A single step in the fast Walsh-Hadamard transform."""
      x_shape = x.shape.as_list()
      x = tf.reshape(x, [-1, 2])  # Reshape so that we have a matrix.
      x = tf.matmul(x, h_core)  # Multiply.
      x = tf.reshape(x, [-1, dim // 2, 2])  # Reshape to rank-3.
      x = tf.transpose(x, perm=permutation)  # Swap last two dimensions.
      x.set_shape(x_shape)  # Failed shape inference in tf.while_loop.
      return x

    def _fwht(x, dim, log2):
      x = tf.reshape(x, [-1, 2, dim // 2])
      # The fast Walsh-Hadamard transform.

      i = tf.constant(0)
      c = lambda i, x: tf.less(i, log2)
      b = lambda i, x: [i + 1, _hadamard_step(x, dim)]
      i, x = tf.while_loop(c, b, [i, x])
      return x

    x = tf.cond(
        tf.equal(dim, 1), lambda: tf.identity(x), lambda: _fwht(x, dim, log2))

    x = tf.reshape(x, [-1, dim])
    x /= tf.sqrt(tf.cast(dim, x.dtype))  # Normalize.
    x.set_shape(original_x_shape)  # Failed shape inference after tf.while_loop.
    return x


def _cmwc_random_sequence(num_elements, seed):
  """Implements a version of the Complementary Multiply with Carry algorithm.

  http://en.wikipedia.org/wiki/Multiply-with-carry

  This implementation serves as a purely TensorFlow implementation of a fully
  deterministic source of pseudo-random number sequence. That is given a
  `Tensor` `seed`, this method will output a `Tensor` with `n` elements, that
  will produce the same sequence when evaluated (assuming the same value of the
  `Tensor` `seed`).

  This method is not particularly efficient, does not come with any guarantee of
  the period length, and should be replaced by appropriate alternative in
  TensorFlow 2.x. In a test in general colab runtime, it took ~0.5s to generate
  1 million values.

  Args:
    num_elements: A Python integer. The number of random values to be generated.
    seed: A scalar `Tensor` of type `tf.int64`.

  Returns:
    A `Tensor` of shape `(num_elements)` and dtype tf.float64, containing random
    values in the range `[0, 1)`.
  """
  if not isinstance(num_elements, int):
    raise TypeError('The num_elements argument must be a Python integer.')
  if num_elements <= 0:
    raise ValueError('The num_elements argument must be positive.')
  if not tf.is_tensor(seed) or seed.dtype != tf.int64:
    raise TypeError('The seed argument must be a tf.int64 Tensor.')

  # For better efficiency of tf.while_loop, we generate `parallelism` random
  # sequences in parallel. The specific constant (sqrt(num_elements) / 10) is
  # hand picked after simple benchmarking for large values of num_elements.
  parallelism = int(math.ceil(math.sqrt(num_elements) / 10))
  num_iters = num_elements // parallelism + 1

  # Create constants needed for the algorithm. The constants and notation
  # follows from the above reference.
  a = tf.tile(tf.constant([3636507990], tf.int64), [parallelism])
  b = tf.tile(tf.constant([2**32], tf.int64), [parallelism])
  logb_scalar = tf.constant(32, tf.int64)
  logb = tf.tile([logb_scalar], [parallelism])
  f = tf.tile(tf.constant([0], dtype=tf.int64), [parallelism])
  bits = tf.constant(0, dtype=tf.int64, name='bits')

  # TensorArray used in tf.while_loop for efficiency.
  values = tf.TensorArray(
      dtype=tf.float64, size=num_iters, element_shape=[parallelism])
  # Iteration counter.
  num = tf.constant(0, dtype=tf.int32, name='num')
  # TensorFlow constant to be used at multiple places.
  val_53 = tf.constant(53, tf.int64, name='val_53')

  # Construct initial sequence of seeds.
  # From a single input seed, we construct multiple starting seeds for the
  # sequences to be computed in parallel.
  def next_seed_fn(i, val, q):
    val = val**7 + val**6 + 1  # PRBS7.
    q = q.write(i, val)
    return i + 1, val, q

  q = tf.TensorArray(dtype=tf.int64, size=parallelism, element_shape=())
  _, _, q = tf.while_loop(lambda i, _, __: i < parallelism,
                          next_seed_fn,
                          [tf.constant(0), seed, q])
  c = q = q.stack()

  # The random sequence generation code.
  def cmwc_step(f, bits, q, c, num, values):
    """A single step of the modified CMWC algorithm."""
    t = a * q + c
    c = b - 1 - tf.bitwise.right_shift(t, logb)
    x = q = tf.bitwise.bitwise_and(t, (b - 1))
    f = tf.bitwise.bitwise_or(tf.bitwise.left_shift(f, logb), x)
    if parallelism == 1:
      f.set_shape((1,))  # Correct for failed shape inference.
    bits += logb_scalar
    def add_val(bits, f, values, num):
      new_val = tf.cast(
          tf.bitwise.bitwise_and(f, (2**val_53 - 1)),
          dtype=tf.float64) * (1 / 2**val_53)
      values = values.write(num, new_val)
      f += tf.bitwise.right_shift(f, val_53)
      bits -= val_53
      num += 1
      return bits, f, values, num
    bits, f, values, num = tf.cond(bits >= val_53,
                                   lambda: add_val(bits, f, values, num),
                                   lambda: (bits, f, values, num))
    return f, bits, q, c, num, values

  def condition(f, bits, q, c, num, values):  # pylint: disable=unused-argument
    return num < num_iters

  _, _, _, _, _, values = tf.while_loop(
      condition,
      cmwc_step,
      [f, bits, q, c, num, values],
  )

  values = tf.reshape(values.stack(), [-1])
  # We generated parallelism * num_iters random values. Take a slice of the
  # first num_elements for the requested Tensor.
  values = values[:num_elements]
  values.set_shape((num_elements,))  # Correct for failed shape inference.
  return  values


def random_signs(num_elements, seed, dtype=tf.float32):
  """Returns a Tensor of `num_elements` random +1/-1 values as `dtype`.

  If run twice with the same seeds, it will produce the same pseudorandom
  numbers. The output is consistent across multiple runs on the same hardware
  (and between CPU and GPU), but may change between versions of TensorFlow or
  on non-CPU/GPU hardware.

  If consistency is required, use `random_signs_cmwc` instead.

  Args:
    num_elements: A Python integer. The number of random values to be generated.
    seed: A shape [2] integer Tensor of seeds to the random number generator.
    dtype: The type of the output.

  Returns:
    A Tensor of `num_elements` random +1/-1 values as `dtype`.
  """
  return tf.cast(
      tf.sign(tf.random.stateless_uniform([num_elements], seed) - 0.5), dtype)


def random_floats(num_elements, seed, dtype=tf.float32):
  """Returns a Tensor of `num_elements` random values in [0, 1) as `dtype`.

  If run twice with the same seeds, it will produce the same pseudorandom
  numbers. The output is consistent across multiple runs on the same hardware
  (and between CPU and GPU), but may change between versions of TensorFlow or
  on non-CPU/GPU hardware.

  If consistency is required, use `random_floats_cmwc` instead.

  Args:
    num_elements: A Python integer. The number of random values to be generated.
    seed: A shape [2] integer Tensor of seeds to the random number generator.
    dtype: The type of the output.

  Returns:
    A Tensor of `num_elements` random values in [0, 1) as `dtype`.
  """
  if dtype not in [tf.float32, tf.float64]:
    raise TypeError('Unsupported type: %s. Supported types are tf.float32 and '
                    'tf.float64 values' % dtype)
  return tf.random.stateless_uniform([num_elements], seed, dtype=dtype)


def random_signs_cmwc(num_elements, seed, dtype=tf.float32):
  """Returns a Tensor of `num_elements` random +1/-1 values as `dtype`."""
  return tf.cast(
      tf.sign(_cmwc_random_sequence(num_elements, seed) - 0.5), dtype)


def random_floats_cmwc(num_elements, seed, dtype=tf.float32):
  """Returns a Tensor of `num_elements` random values in [0, 1) as `dtype`."""
  if dtype not in [tf.float32, tf.float64]:
    raise TypeError(
        'Unsupported type: %s. Supported types are tf.float32 and '
        'tf.float64 values' % dtype)
  return tf.cast(_cmwc_random_sequence(num_elements, seed), dtype)


def pack_into_int(value, input_bitrange, target_bitrange):
  """Pack integers in range [0, 2**`input_bitrange`-1] into integer values.

  This utility simply concatenates the relevant bits of the input values into
  a sequence of integer values.

  The `target_bitrange` can be used to not use all bits of the return type.
  This can be useful for instance when the resulting values can be serialized as
  a varint. In such case, using only 7 bits per byte could be more desirable.

  NOTE: This only uses basic math operations to implement the bit manipulation,
  not any bitwise operations, which is relevant in environments where only a
  subset of TensorFlow ops/kernels are available. Moreover, if values outside of
  the expected range are provided at runtime, an error will *not* be raised.
  The behavior of this method in such case is undefined.

  Args:
    value: An integer Tensor to be packed.
    input_bitrange: An integer. The number of relevant bits in `value`.
    target_bitrange: An integer. The number of bits to be used in packed
      representation.

  Returns:
    An integer Tensor representing `value` of the same dtype as `value`.
  """
  # TODO(b/161433177): Provide a general solution without memory overhead.
  # Special cases implemented without extra memory overhead.
  if input_bitrange == 6 and target_bitrange == 28:
    return _pack_into_int_6_28(value)
  if input_bitrange == 7 and target_bitrange == 28:
    return _pack_into_int_7_28(value)
  if input_bitrange == 8 and target_bitrange == 28:
    return _pack_into_int_8_28(value)
  if input_bitrange == 12 and target_bitrange == 28:
    return _pack_into_int_12_28(value)

  # General solution with possible extra memory overhead.
  logging.warning('This code path can temporarily allocate extra memory. If '
                  'memory footprint is a problem, consider different bitpacking'
                  ' method or turning this functionality off. See b/161433177')
  if input_bitrange > 1:
    value = tf.reshape(value, [-1, 1])
    value = _expand_to_binary_form(value, input_bitrange)
  return _pack_binary_form(value, target_bitrange)


def unpack_from_int(value, original_bitrange, target_bitrange, shape):
  """Unpack integers into the range of [0, 2**`original_bitrange`-1].

  This utility is to be used as the inverse of `pack_into_int` utility.

  The shape of the original input is needed for uniqueness of the inverse
  operation -- inputs of different shapes can be packed into the same
  representation.

  Args:
    value: An integer Tensor to be unpacked.
    original_bitrange: An integer. The number of bits used in the original
      representation.
    target_bitrange: An integer. The number of bits used in the packed
      representation.
    shape: The shape of the original input.

  Returns:
    An integer Tensor representing the unpacked `value` of the same dtype as
    `value`.
  """
  # TODO(b/161433177): Provide a general solution without memory overhead.
  # Special cases implemented without extra memory overhead.
  if original_bitrange == 6 and target_bitrange == 28:
    return _unpack_from_int_6_28(value, shape)
  if original_bitrange == 7 and target_bitrange == 28:
    return _unpack_from_int_7_28(value, shape)
  if original_bitrange == 8 and target_bitrange == 28:
    return _unpack_from_int_8_28(value, shape)
  if original_bitrange == 12 and target_bitrange == 28:
    return _unpack_from_int_12_28(value, shape)

  # General solution with extra memory overhead.
  logging.warning('This code path can temporarily allocate extra memory. If '
                  'memory footprint is a problem, consider different bitpacking'
                  ' method or turning this functionality off. See b/161433177')
  value = _expand_to_binary_form(value, target_bitrange)
  value = tf.slice(value, [0], [tf.reduce_prod(shape) * original_bitrange])
  if original_bitrange > 1:
    return tf.reshape(_pack_binary_form(value, original_bitrange), shape)
  else:
    return tf.reshape(value, shape)


def _pack_binary_form(value, target_bits):
  # Reshape the binary input to have target_bits columns, padding with zeros if
  # necessary to fit the dimension. The bitpacked representation is computed
  # as product with vector [1, 2, 4, ..., 2**target_bits].
  packing_vector = tf.constant([[2**i] for i in range(target_bits)],
                               value.dtype)
  extra_zeros = tf.zeros(
      tf.math.mod(-tf.shape(value), target_bits), value.dtype)
  reshaped_x = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, target_bits])
  return tf.matmul(reshaped_x, packing_vector)


def _expand_to_binary_form(value, input_bits):
  # This operation is inverse of _pack_binary_form, except padded zeros are not
  # removed.
  expand_vector = tf.constant([2**i for i in range(input_bits)], value.dtype)
  bits = tf.math.mod(tf.math.floordiv(value, expand_vector), 2)
  return tf.reshape(bits, [-1])


def _pack_into_int_6_28(value):
  """Implementation of `pack_into_int` for specific bitranges.

  This method corresponts to `(input_bitrange, target_bitrange)` form the
  `pack_into_int` method equal to `(6, 28)`. This method relies on the fact that
  14 values in 6-bit bitrange can be packed into 3 values in 28-bitrange
  (14 = least_common_multiple(6, 28) / 6).

  It reshapes the input into matrix of 14 columns and performs operations on the
  columns of the matrix, thus vectorizing the operations and avoiding memory
  overhead of an earlier general implementation.

  Args:
    value: An integer Tensor to be packed with values in [0, 2**6 - 1].

  Returns:
    An integer Tensor representing `value` of the same dtype as `value`.
  """
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 14), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 14])

  a = (val[:, 0] +
       val[:, 1] * 2**6 +
       val[:, 2] * 2**12 +
       val[:, 3] * 2**18 +
       tf.math.mod(val[:, 4], 2**4) * 2**24)
  b = (tf.math.floordiv(val[:, 4], 2**4) +
       val[:, 5] * 2**2 +
       val[:, 6] * 2**8 +
       val[:, 7] * 2**14 +
       val[:, 8] * 2**20 +
       tf.math.mod(val[:, 9], 2**2) * 2**26)
  c = (tf.math.floordiv(val[:, 9], 2**2) +
       val[:, 10] * 2**4 +
       val[:, 11] * 2**10 +
       val[:, 12] * 2**16 +
       val[:, 13] * 2**22)

  packed_val = tf.reshape(tf.stack([a, b, c], 1), [-1, 1])
  if extra_zeros.shape[0] in [5, 6, 7, 8, 9]:
    # We added unnecessary product of zeros to the representation.
    packed_val = tf.slice(packed_val, [0, 0], [packed_val.shape[0] - 1, 1])
  if extra_zeros.shape[0] in [10, 11, 12, 13]:
    # We added unnecessary two products of zeros to the representation.
    packed_val = tf.slice(packed_val, [0, 0], [packed_val.shape[0] - 2, 1])
  return packed_val


def _pack_into_int_7_28(value):
  """Implementation of `pack_into_int` for specific bitranges.

  This method corresponts to `(input_bitrange, target_bitrange)` form the
  `pack_into_int` method equal to `(7, 28)`. This method relies on the fact that
  4 values in 7-bit bitrange can be packed into 1 value in 28-bitrange
  (4 = least_common_multiple(7, 28) / 7).

  It reshapes the input into matrix of 4 columns and performs operations on the
  columns of the matrix, thus vectorizing the operations and avoiding memory
  overhead of an earlier general implementation.

  Args:
    value: An integer Tensor to be packed with values in [0, 2**7 - 1].

  Returns:
    An integer Tensor representing `value` of the same dtype as `value`.
  """
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 4), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 4])

  packed_val = (val[:, 0] +
                val[:, 1] * 2**7 +
                val[:, 2] * 2**14 +
                val[:, 3] * 2**21)

  return tf.reshape(packed_val, [-1, 1])


def _pack_into_int_8_28(value):
  """Implementation of `pack_into_int` for specific bitranges.

  This method corresponts to `(input_bitrange, target_bitrange)` form the
  `pack_into_int` method equal to `(8, 28)`. This method relies on the fact that
  7 values in 8-bit bitrange can be packed into 2 values in 28-bitrange
  (7 = least_common_multiple(8, 28) / 8).

  It reshapes the input into matrix of 7 columns and performs operations on the
  columns of the matrix, thus vectorizing the operations and avoiding memory
  overhead of an earlier general implementation.

  Args:
    value: An integer Tensor to be packed with values in [0, 2**8 - 1].

  Returns:
    An integer Tensor representing `value` of the same dtype as `value`.
  """
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 7), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 7])

  a = (val[:, 0] +
       val[:, 1] * 2**8 +
       val[:, 2] * 2**16 +
       tf.math.mod(val[:, 3], 2**4) * 2**24)
  b = (tf.math.floordiv(val[:, 3], 2**4) +
       val[:, 4] * 2**4 +
       val[:, 5] * 2**12 +
       val[:, 6] * 2**20)

  packed_val = tf.reshape(tf.stack([a, b], 1), [-1, 1])
  if extra_zeros.shape[0] in [4, 5, 6]:
    # We added unnecessary sum of zeros to the representation.
    packed_val = tf.slice(packed_val, [0, 0], [packed_val.shape[0] - 1, 1])
  return packed_val


def _pack_into_int_12_28(value):
  """Implementation of `pack_into_int` for specific bitranges.

  This method corresponts to `(input_bitrange, target_bitrange)` form the
  `pack_into_int` method equal to `(12, 28)`. This method relies on the fact
  that 7 values in 12-bit bitrange can be packed into 3 values in 28-bitrange
  (7 = least_common_multiple(12, 28) / 12).

  It reshapes the input into matrix of 7 columns and performs operations on the
  columns of the matrix, thus vectorizing the operations and avoiding memory
  overhead of an earlier general implementation.

  Args:
    value: An integer Tensor to be packed with values in [0, 2**8 - 1].

  Returns:
    An integer Tensor representing `value` of the same dtype as `value`.
  """
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 7), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 7])

  a = (val[:, 0] +
       val[:, 1] * 2**12 +
       tf.math.mod(val[:, 2], 2**4) * 2**24)
  b = (tf.math.floordiv(val[:, 2], 2**4) +
       val[:, 3] * 2**8 +
       tf.math.mod(val[:, 4], 2**8) * 2**20)
  c = (tf.math.floordiv(val[:, 4], 2**8) +
       val[:, 5] * 2**4 +
       val[:, 6] * 2**16)

  packed_val = tf.reshape(tf.stack([a, b, c], 1), [-1, 1])
  if extra_zeros.shape[0] in [3, 4]:
    # We added unnecessary sum of zeros to the representation.
    packed_val = tf.slice(packed_val, [0, 0], [packed_val.shape[0] - 1, 1])
  if extra_zeros.shape[0] in [5, 6]:
    # We added unnecessary two sums of zeros to the representation.
    packed_val = tf.slice(packed_val, [0, 0], [packed_val.shape[0] - 2, 1])
  return packed_val


def _unpack_from_int_6_28(value, shape):
  """Inverse operation of `_pack_into_int_6_28`."""
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 3), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 3])

  a = tf.math.mod(val[:, 0], 2**6)
  b = tf.math.mod(tf.math.floordiv(val[:, 0], 2**6), 2**6)
  c = tf.math.mod(tf.math.floordiv(val[:, 0], 2**12), 2**6)
  d = tf.math.mod(tf.math.floordiv(val[:, 0], 2**18), 2**6)
  e = tf.math.floordiv(val[:, 0], 2**24) + tf.math.mod(val[:, 1], 2**2) * 2**4
  f = tf.math.mod(tf.math.floordiv(val[:, 1], 2**2), 2**6)
  g = tf.math.mod(tf.math.floordiv(val[:, 1], 2**8), 2**6)
  h = tf.math.mod(tf.math.floordiv(val[:, 1], 2**14), 2**6)
  i = tf.math.mod(tf.math.floordiv(val[:, 1], 2**20), 2**6)
  j = tf.math.floordiv(val[:, 1], 2**26) + tf.math.mod(val[:, 2], 2**4) * 2**2
  k = tf.math.mod(tf.math.floordiv(val[:, 2], 2**4), 2**6)
  l = tf.math.mod(tf.math.floordiv(val[:, 2], 2**10), 2**6)
  m = tf.math.mod(tf.math.floordiv(val[:, 2], 2**16), 2**6)
  n = tf.math.mod(tf.math.floordiv(val[:, 2], 2**22), 2**6)

  unpacked_val = tf.reshape(
      tf.stack([a, b, c, d, e, f, g, h, i, j, k, l, m, n], 1), [-1,])
  unpacked_val = tf.slice(unpacked_val, [0], [tf.reduce_prod(shape)])
  return tf.reshape(unpacked_val, shape)


def _unpack_from_int_7_28(value, shape):
  """Inverse operation of `_pack_into_int_7_28`."""
  val = tf.reshape(value, [-1, 1])

  a = tf.math.mod(val[:, 0], 2**7)
  b = tf.math.mod(tf.math.floordiv(val[:, 0], 2**7), 2**7)
  c = tf.math.mod(tf.math.floordiv(val[:, 0], 2**14), 2**7)
  d = tf.math.mod(tf.math.floordiv(val[:, 0], 2**21), 2**7)

  unpacked_val = tf.reshape(tf.stack([a, b, c, d], 1), [-1,])
  unpacked_val = tf.slice(unpacked_val, [0], [tf.reduce_prod(shape)])
  return tf.reshape(unpacked_val, shape)


def _unpack_from_int_8_28(value, shape):
  """Inverse operation of `_pack_into_int_8_28`."""
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 2), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 2])

  a = tf.math.mod(val[:, 0], 2**8)
  b = tf.math.mod(tf.math.floordiv(val[:, 0], 2**8), 2**8)
  c = tf.math.mod(tf.math.floordiv(val[:, 0], 2**16), 2**8)
  d = tf.math.floordiv(val[:, 0], 2**24) + tf.math.mod(val[:, 1], 2**4) * 2**4
  e = tf.math.mod(tf.math.floordiv(val[:, 1], 2**4), 2**8)
  f = tf.math.mod(tf.math.floordiv(val[:, 1], 2**12), 2**8)
  g = tf.math.floordiv(val[:, 1], 2**20)

  unpacked_val = tf.reshape(tf.stack([a, b, c, d, e, f, g], 1), [-1,])
  unpacked_val = tf.slice(unpacked_val, [0], [tf.reduce_prod(shape)])
  return tf.reshape(unpacked_val, shape)


def _unpack_from_int_12_28(value, shape):
  """Inverse operation of `_pack_into_int_12_28`."""
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 3), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 3])

  a = tf.math.mod(val[:, 0], 2**12)
  b = tf.math.mod(tf.math.floordiv(val[:, 0], 2**12), 2**12)
  c = tf.math.floordiv(val[:, 0], 2**24) + tf.math.mod(val[:, 1], 2**8) * 2**4
  d = tf.math.mod(tf.math.floordiv(val[:, 1], 2**8), 2**12)
  e = tf.math.floordiv(val[:, 1], 2**20) + tf.math.mod(val[:, 2], 2**4) * 2**8
  f = tf.math.mod(tf.math.floordiv(val[:, 2], 2**4), 2**12)
  g = tf.math.mod(tf.math.floordiv(val[:, 2], 2**16), 2**12)

  unpacked_val = tf.reshape(tf.stack([a, b, c, d, e, f, g], 1), [-1,])
  unpacked_val = tf.slice(unpacked_val, [0], [tf.reduce_prod(shape)])
  return tf.reshape(unpacked_val, shape)
