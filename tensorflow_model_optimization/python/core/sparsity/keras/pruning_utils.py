# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for adding pruning related ops to the graph.

"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import g3
import numpy as np
import tensorflow as tf


def kronecker_product(mat1, mat2):
  """Computes the Kronecker product of two matrices mat1 and mat2.

  Args:
    mat1: A matrix of size m x n
    mat2: A matrix of size p x q

  Returns:
    Kronecker product of matrices mat1 and mat2 of size mp x nq
  """

  m1, n1 = mat1.get_shape().as_list()
  mat1_rsh = tf.reshape(mat1, [m1, 1, n1, 1])
  m2, n2 = mat2.get_shape().as_list()
  mat2_rsh = tf.reshape(mat2, [1, m2, 1, n2])
  return tf.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])


def expand_tensor(tensor, block_size):
  """Expands a 2D tensor by replicating the tensor values.

  This is equivalent to the kronecker product of the tensor and a matrix of
  ones of size block_size.

  Example:

  tensor = [[1,2]
            [3,4]]
  block_size = [2,2]

  result = [[1 1 2 2]
            [1 1 2 2]
            [3 3 4 4]
            [3 3 4 4]]

  Args:
    tensor: A 2D tensor that needs to be expanded.
    block_size: List of integers specifying the expansion factor.

  Returns:
    The expanded tensor

  Raises:
    ValueError: if tensor is not rank-2 or block_size is does not have 2
    elements.
  """
  if tensor.get_shape().ndims != 2:
    raise ValueError("Input tensor must be rank 2")

  if len(block_size) != 2:
    raise ValueError("block_size must have 2 elements")

  block_height, block_width = block_size

  def _tile_rows(tensor, multiple):
    """Create a new tensor by tiling the tensor along rows."""
    return tf.tile(tensor, [multiple, 1])

  def _generate_indices(num_rows, block_dim):
    indices = np.zeros(shape=[num_rows * block_dim, 1], dtype=np.int32)
    for k in range(block_dim):
      for r in range(num_rows):
        indices[k * num_rows + r] = r * block_dim + k
    return indices

  def _replicate_rows(tensor, multiple):
    tensor_shape = tensor.shape.as_list()
    expanded_shape = [tensor_shape[0] * multiple, tensor_shape[1]]
    indices = tf.constant(_generate_indices(tensor_shape[0], multiple))
    return tf.scatter_nd(indices, _tile_rows(tensor, multiple), expanded_shape)

  expanded_tensor = tensor

  # Expand rows by factor block_height.
  if block_height > 1:
    expanded_tensor = _replicate_rows(tensor, block_height)

  # Transpose and expand by factor block_width. Transpose the result.
  if block_width > 1:
    expanded_tensor = tf.transpose(
        _replicate_rows(tf.transpose(expanded_tensor), block_width))

  return expanded_tensor


def factorized_pool(input_tensor,
                    window_shape,
                    pooling_type,
                    strides,
                    padding,
                    name=None):
  """Performs m x n pooling through a combination of 1xm and 1xn pooling.

  Args:
    input_tensor: Input tensor. Must be rank 2
    window_shape: Pooling window shape
    pooling_type: Either 'MAX' or 'AVG'
    strides: The stride of the pooling window
    padding: 'SAME' or 'VALID'.
    name: Name of the op

  Returns:
    A rank 2 tensor containing the pooled output

  Raises:
    ValueError: if the input tensor is not rank 2
  """
  if input_tensor.get_shape().ndims != 2:
    raise ValueError("factorized_pool() accepts tensors of rank 2 only")

  [height, width] = input_tensor.get_shape()
  if name is None:
    name = "factorized_pool"
  with tf.name_scope(name):
    input_tensor_aligned = tf.reshape(input_tensor, [1, 1, height, width])

    height_pooling = tf.nn.pool(
        input_tensor_aligned,
        window_shape=[1, window_shape[0]],
        pooling_type=pooling_type,
        strides=[1, strides[0]],
        padding=padding)
    swap_height_width = tf.transpose(height_pooling, perm=[0, 1, 3, 2])

    width_pooling = tf.nn.pool(
        swap_height_width,
        window_shape=[1, window_shape[1]],
        pooling_type=pooling_type,
        strides=[1, strides[1]],
        padding=padding)

  return tf.squeeze(tf.transpose(width_pooling, perm=[0, 1, 3, 2]))


def convert_to_tuple_of_two_int(value, name):
  """Transforms iterable of 2 integers into an tuple of 2 integers.

  Args:
    value: A iterable of 2 ints.
    name: The name of the argument being validated, e.g., sparsity_m_by_n.

  Returns:
    A tuple of 2 integers.

  Raises:
    ValueError: If something else than an iterable of ints was passed.
  """
  try:
    value_tuple = tuple(value)
  except TypeError:
    raise ValueError(f"The {name} argument must be a tuple/list of 2 integers."
                     f"received: {str(value)}.") from None
  if len(value_tuple) != 2:
    raise ValueError(f"The {name} argument must be a tuple/list of 2 integers."
                     f"received: {str(value)}.")
  for single_value in value_tuple:
    if not isinstance(single_value, int):
      raise ValueError(
          f"The {name} argument must be a tuple/list of 2 integers."
          f"received: {str(value)} including element {str(single_value)} "
          f"of type {str(type(single_value))}.")

  return value_tuple


def weights_rearrange(weights):
  """Rearrange weights tensor so that m by n sparsity structure applied in last channel.

  This is a m_by_n sparsity helper function.

  m by n sparsity: every group of consecutive n values contains
    at least m zeros on the last channel in TFLite data format.

  * In case of Conv2D weights:
      TF data format is [height, width, channel_in, channel_out],
      TFLite data format is [channel_out, height, width, channel_in]
    Rearranged Conv2D weights format: [channel_out x height x width, channel_in]

  * In case of Dense weights:
      TF data format is [channel_in, channel_out],
      TFLite data format is [width, channel_in]
    Rearranged Conv2D weights format: [width, channel_in]

  Args:
    weights: weights tensor. Must be rank 2 or 4.

  Returns:
    A rank 2 weight tensor.

  Raises:
    ValueError: if the input tensor is not rank 2 or 4.
  """
  if weights.shape.rank == 2:
    prepared_weights = tf.transpose(weights)
  elif weights.shape.rank == 4:
    perm_weights = tf.transpose(weights, perm=[3, 0, 1, 2])
    prepared_weights = tf.reshape(perm_weights,
                                  [tf.reduce_prod(perm_weights.shape[:-1]), -1])
  else:
    raise ValueError(
        f"weight tensor with shape: {weights.shape} is not supported.")

  return prepared_weights


def m_by_n_sparsity_mask_prepare(mask, weights_shape):
  """Reshape and permute sparsity mask, so that it match original weights data format.

  This is a m_by_n sparsity helper function.

  Args:
    mask: A 2-D tensor. Must be rank 2 or 4.
    weights_shape: shape of weights

  Returns:
    A sparsity mask that matches weights data format.

  Raises:
    ValueError:
      if the input tensor is not rank 2 or 4.
    InvalidArgumentError:
      if number of elements mismatch between mask and weights,
      if shape of prepared mask mismatch shape of weights.
  """
  tf.debugging.assert_equal(
      tf.size(mask),
      tf.reduce_prod(weights_shape),
      message="number of elements mismatch between mask and weights.",
  )

  if mask.shape.rank != 2:
    raise ValueError(f"rank of mask(rank:{mask.shape.rank}) should be 2.")

  if weights_shape.rank == 2:
    prepared_mask = tf.transpose(mask)
  elif weights_shape.rank == 4:
    reshaped_mask = tf.reshape(
        mask,
        [
            weights_shape[-1], weights_shape[0], weights_shape[1],
            weights_shape[2]
        ],
    )
    prepared_mask = tf.transpose(reshaped_mask, perm=[1, 2, 3, 0])
  else:
    raise ValueError(
        f"weight tensor with shape: {weights_shape} is not supported.")

  tf.debugging.assert_equal(
      prepared_mask.shape,
      weights_shape,
      message="shape of prepared mask mismatch shape of weights.")

  return prepared_mask


def generate_m_by_n_mask(weights, m_by_n=(2, 4)):
  """Generate m-by-n sparsity mask.

  This is a m_by_n sparsity helper function.

  Args:
    weights: a rank 2 tensor.
    m_by_n: a tuple of 2 integers (m, n), indicates m zeros in every n
      consecutive values, m must be smaller than n. Default to (2, 4).

  Returns:
    A rank 2 m-by-n sparsity mask.

  Raises:
    InvalidArgumentError: if m not smaller than n.
  """
  num_zeros, block_size = tf.constant(m_by_n[0]), tf.constant(m_by_n[1])
  tf.debugging.assert_less(
      num_zeros,
      block_size,
      message=f"Argument m_by_n received {m_by_n}, m be must smaller than n.")
  num_non_zeros = block_size - num_zeros
  abs_weights = tf.abs(weights)

  # add zero-padding
  pad_after = block_size - abs_weights.shape[-1] % block_size
  abs_weights_pad = tf.pad(abs_weights, [[0, 0], [0, pad_after]], "CONSTANT")

  num_blocks = tf.size(abs_weights_pad) // block_size
  reshaped_weights_into_blocks = tf.reshape(abs_weights_pad,
                                            [num_blocks, block_size])
  _, top_k_indices = tf.math.top_k(
      reshaped_weights_into_blocks, k=num_non_zeros, sorted=False)
  ind_i, _ = tf.meshgrid(
      tf.range(num_blocks), tf.range(num_non_zeros), indexing="ij")
  ind_ij = tf.stack([ind_i, top_k_indices], axis=-1)
  sparsity_mask_pad = tf.scatter_nd(ind_ij, tf.ones([num_blocks,
                                                     num_non_zeros]),
                                    [num_blocks, block_size])
  reshaped_sparsity_mask_pad = tf.reshape(sparsity_mask_pad,
                                          tf.shape(abs_weights_pad))

  # remove padding from mask
  sparsity_mask = tf.slice(reshaped_sparsity_mask_pad, [0, 0],
                           abs_weights.shape)

  return sparsity_mask


def is_pruned_m_by_n(weights, m_by_n=(2, 4), last_channel: str = "C_OUT"):
  """Check m by n sparsity pattern on Weight Tensor.

  This is a m_by_n sparsity helper function.

  Args:
    weights: A tensor of layer weights.
    m_by_n: a tuple of 2 integers (m, n), indicates m zeros in every n
      consecutive values, m must be smaller than n. Default to (2, 4).
    last_channel: A string, 'C_OUT'(default) and 'C_IN' are supported.  Last
      channel of weights tensor.
        Conv2D weights in TF: [H, W, C_IN, C_OUT], TFLite: [C_OUT, H, W, C_IN]
        DENSE weights in TF: [C_IN, C_OUT], TFLite: [C_OUT, C_IN]

  Returns:
    A boolean value: True if weights are pruned with sparsity m_by_n
      on the last channel.

  Raises:
    ValueError:
      if unsupported last_channel.
      if m is larger than n.

  """
  num_zeros, num_elem = m_by_n
  if num_zeros > num_elem:
    raise ValueError(f"number of zeros can't be more than number elements. "
                     f"received: {num_zeros} zeros in {num_elem} elements.")
  num_non_zeros = num_elem - num_zeros

  if last_channel.endswith("C_IN"):
    prepared_weights = tf.reshape(weights,
                                  [tf.reduce_prod(weights.shape[:-1]), -1])
  elif last_channel.endswith("C_OUT"):
    prepared_weights = weights_rearrange(weights)
  else:
    raise ValueError("last_channel must be `C_IN` or `C_OUT`")

  prepared_weights_np = prepared_weights.numpy()
  for row in range(0, prepared_weights_np.shape[0]):
    for col in range(0, prepared_weights_np.shape[1], num_elem):
      if (np.count_nonzero(prepared_weights_np[row, col:col + num_elem]) >
          num_non_zeros):
        return False
  return True
