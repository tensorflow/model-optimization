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
"""A collection of common encoders.

Most users of the `tensor_encoding` package should only need to access symbols
in this file, unless a specific advanced functionality is needed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import core_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import gather_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import simple_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages import stages_impl


def as_simple_encoder(encoder, tensorspec):
  """Wraps an `Encoder` object as a `SimpleEncoder`.

  Args:
    encoder: An `Encoder` object to be used to encoding.
    tensorspec: A `TensorSpec`. The created `SimpleEncoder` will be constrained
      to only encode input values compatible with `tensorspec`.

  Returns:
    A `SimpleEncoder`.

  Raises:
    TypeError:
      If `encoder` is not an `Encoder` or `tensorspec` is not a `TensorSpec`.
  """
  if not isinstance(encoder, core_encoder.Encoder):
    raise TypeError('The encoder must be an instance of `Encoder`.')
  if not isinstance(tensorspec, tf.TensorSpec):
    raise TypeError('The tensorspec must be a tf.TensorSpec.')
  return simple_encoder.SimpleEncoder(encoder, tensorspec)


def as_gather_encoder(encoder, tensorspec):
  """Wraps an `Encoder` object as a `GatherEncoder`.

  Args:
    encoder: An `Encoder` object to be used to encoding.
    tensorspec: A `TensorSpec`. The created `GatherEncoder` will be constrained
      to only encode input values compatible with `tensorspec`.

  Returns:
    A `GatherEncoder`.

  Raises:
    TypeError:
      If `encoder` is not an `Encoder` or `tensorspec` is not a `TensorSpec`.
  """
  if not isinstance(encoder, core_encoder.Encoder):
    raise TypeError('The encoder must be an instance of `Encoder`.')
  if not isinstance(tensorspec, tf.TensorSpec):
    raise TypeError('The tensorspec must be a tf.TensorSpec.')
  return gather_encoder.GatherEncoder.from_encoder(encoder, tensorspec)


def identity():
  """Returns identity `Encoder`."""
  return core_encoder.EncoderComposer(
      stages_impl.IdentityEncodingStage()).make()


def uniform_quantization(bits, **kwargs):
  """Returns uniform quanitzation `Encoder`.

  The `Encoder` first reshapes the input to a rank-1 `Tensor`, then applies
  uniform quantization with the extreme values being the minimum and maximum of
  the vector being encoded. Finally, the quantized values are bitpacked to an
  integer type.

  The `Encoder` is a composition of the following encoding stages:
  * `FlattenEncodingStage`
  * `UniformQuantizationEncodingStage`
  * `BitpackingEncodingStage`

  Args:
    bits: Number of bits to quantize into.
    **kwargs: Keyword arguments.

  Returns:
    The quantization `Encoder`.
  """
  return core_encoder.EncoderComposer(
      stages_impl.BitpackingEncodingStage(bits)).add_parent(
          stages_impl.UniformQuantizationEncodingStage(bits, **kwargs),
          stages_impl.UniformQuantizationEncodingStage.ENCODED_VALUES_KEY
      ).add_parent(stages_impl.FlattenEncodingStage(),
                   stages_impl.FlattenEncodingStage.ENCODED_VALUES_KEY).make()


def hadamard_quantization(bits):
  """Returns hadamard quanitzation `Encoder`.

  The `Encoder` first reshapes the input to a rank-1 `Tensor`, and applies the
  Hadamard transform (rotation). It then applies uniform quantization with the
  extreme values being the minimum and maximum of the rotated vector being
  encoded. Finally, the quantized values are bitpacked to an integer type.

  The `Encoder` is a composition of the following encoding stages:
  * `FlattenEncodingStage` - reshaping the input to a vector.
  * `HadamardEncodingStage` - applying the Hadamard transform.
  * `UniformQuantizationEncodingStage` - applying uniform quantization.
  * `BitpackingEncodingStage` - bitpacking the result into integer values.

  Args:
    bits: Number of bits to quantize into.

  Returns:
    The hadamard quantization `Encoder`.
  """
  return core_encoder.EncoderComposer(
      stages_impl.BitpackingEncodingStage(bits)).add_parent(
          stages_impl.UniformQuantizationEncodingStage(bits), stages_impl
          .UniformQuantizationEncodingStage.ENCODED_VALUES_KEY).add_parent(
              stages_impl.HadamardEncodingStage(),
              stages_impl.HadamardEncodingStage.ENCODED_VALUES_KEY).add_parent(
                  stages_impl.FlattenEncodingStage(),
                  stages_impl.FlattenEncodingStage.ENCODED_VALUES_KEY).make()


def drive(bias_correction=True):
  """Returns DRIVE `Encoder`.

  First, the `Encoder` reshapes the input to a rank-1 `Tensor` and applies a
  randomized Hadamard transform (rotation). It then applies a rotation-aware
  sign, and, finally, the quantized values are bit-packed into an integer type.

  This encoder is derived from the source published with "DRIVE: One-bit
  Distributed Mean Estimation" (NeurIPS '21;
  https://arxiv.org/pdf/2105.08339.pdf), and the algorithm presented therein.

  Limitations:
  (1) In the implementation of HadamardEncodingStage a single seed is shared
  among senders, as described in the paper this should be used when the number
  of senders are no more than log of the dimension of the input tensor.
  (2) This encoder works better on larger tensors. An ideal preprocessing
  stage would concatenate the input model into a single tensor. Additionally,
  the ability to mark a few tensors for being skipped would also be helpful
  (e.g., normalization layers). Currently, this is not always possible with
  the tensor encoders API.

  Despite the limitations of this implementation, this achieves accuracy similar
  to sending the full tensors for many distributed learning scenarios.

  The `Encoder` is a composition of the following encoding stages:
  * `FlattenEncodingStage` - reshaping the input tensor into a vector.
  * `HadamardEncodingStage` - applying the Hadamard transform.
  * `RotationAwareSignEncodingStage` - applying a rotation-aware sign.
  * `BitpackingEncodingStage` - bit-packing the result into integer values.

  Args:
    bias_correction: A Python bool, whether to use bias correcting or
      MSE minimizing scale.
      If `True`, the encoding is unbiased on expectation.
      If `False`, the encoding minimizes the MSE.

  Returns:
    The DRIVE `Encoder`.
  """
  return core_encoder.EncoderComposer(
    stages_impl.BitpackingEncodingStage(1)).add_parent(
        stages_impl.RotationAwareSignEncodingStage(bias_correction), stages_impl
        .RotationAwareSignEncodingStage.ENCODED_VALUES_KEY).add_parent(
            stages_impl.HadamardEncodingStage(),
            stages_impl.HadamardEncodingStage.ENCODED_VALUES_KEY).add_parent(
                stages_impl.FlattenEncodingStage(),
                stages_impl.FlattenEncodingStage.ENCODED_VALUES_KEY).make()
