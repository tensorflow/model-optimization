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
import logging
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
    raise ValueError('Input tensor must be rank 2')

  if len(block_size) != 2:
    raise ValueError('block_size must have 2 elements')

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
    raise ValueError('factorized_pool() accepts tensors of rank 2 only')

  [height, width] = input_tensor.get_shape()
  if name is None:
    name = 'factorized_pool'
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

def check_if_applicable_sparsity_2x4(weight):
  """ This function checks that the sparsity 2x4 could be applied to weight.

  The sparsity 2x4 is applied to Conv2D layer and Dense. We assume that
  the number of channels is divisible by four in case of Conv2D or the width is
  divisible by four for Dense. If the condition is not satisfied, we fallback
  to the unstructured pruning.

  Returns:
    Boolean that indicates whether sparsity 2x4 is applicable.
  """
  if (tf.convert_to_tensor(weight).get_shape()[-1] % 4) != 0 :
    return False
  return True


def generate_m_to_n_mask(weights, window_shape, k):
  """Generates mask for the given weights with the given window_shape.
    For example, in case of sparsity 2x4, the window_shape is expected
    to be 1x4 and k=2. We don't apply any padding, because we assume
    that the given weights don't require it for the given window shape.
    We do check on this before this function. We set zeros to 2 out of
    4 elements with the smallest absolute value using top_k function.

    Args:
      weights: Input weights tensor.
      window_shape: Pooling window shape.
      k: How many elements should be set to zero.

    Returns:
      The generated mask for the given weights.
      If the returned mask is None, we assume that something went wrong
      and we fallback to the unstructured pruning.

    Raises:
      Does not raise any error.
  """
  abs_weights = tf.abs(weights)
  abs_weights_shape = abs_weights.get_shape()

  # Sanity check: height or width should be bigger than window_shape
  if (abs_weights_shape[0] < window_shape[0] or \
    abs_weights_shape[1] < window_shape[1]):
    logging.warning('We cannot apply sparsity MxN, '
                  'because weights size is too small.')
    return None

  if (not hasattr(abs_weights, 'numpy')):
    logging.warning('We cannot apply sparsity MxN, '
                  'weights do not have any values.')
    return None

  # Reshape weights into blocks, so we can apply top_k function.
  # Note that it works only on inner-most dimension.
  flatten_weights = abs_weights.numpy().reshape(-1)

  logging.info('We are applying {}x{} sparsity for {} parameters'\
    .format(k, window_shape[1], tf.size(weights)))

  number_of_blocks = len(flatten_weights) // (window_shape[0] * window_shape[1])
  reshaped_weights_into_blocks = tf.reshape(flatten_weights,
    [number_of_blocks, window_shape[0], window_shape[1]])

  # Apply top_k to find indices of elements that will be nullified.
  _, top_k_indices = tf.math.top_k(reshaped_weights_into_blocks, k=k, sorted=False)

  # Reconstruct full indices of the top_k_indices
  shape_of_reshaped_weights = reshaped_weights_into_blocks.get_shape().as_list()

  dim_00, dim_11, _ = tf.meshgrid(
    tf.range(shape_of_reshaped_weights[0]), tf.range(shape_of_reshaped_weights[1]),
    tf.range(k), indexing='ij')

  updates = tf.ones([shape_of_reshaped_weights[0], shape_of_reshaped_weights[1], k],
    dtype=tf.float32)
  index = tf.stack([dim_00, dim_11, top_k_indices], axis=-1)
  top_k_mask_4d = tf.scatter_nd(index, updates,
    tf.shape(reshaped_weights_into_blocks))

  # Convert back to the shape of weights
  top_k_mask_result = tf.reshape(top_k_mask_4d, tf.shape(abs_weights))

  return top_k_mask_result

def is_pruned_2x4(weights):
  """Returns true if weights are pruned with sparsity 2x4,
  otherwise false.

  Args:
      weights: Input weights tensor.
  """
  weights = tf.abs(weights)
  flatten_weights = weights.numpy().reshape(-1)
  for i in range(0, len(flatten_weights), 4):
    if np.count_nonzero(flatten_weights[i:i+4]) > 2:
      return False
  return True