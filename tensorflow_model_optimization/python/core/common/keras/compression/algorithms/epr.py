# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Entropy Penalized Reparameterization algorithm.

This is an implementation of the method described in:
> "Scalable Model Compression by Entropy Penalized Reparameterization"<br />
> D. Oktay, J. Ballé, S. Singh, A. Shrivastava<br />
> https://arxiv.org/abs/1906.06624
"""

import functools
from typing import List
import tensorflow as tf
import tensorflow_compression as tfc
from tensorflow_model_optimization.python.core.common.keras.compression import algorithm


class EPR(algorithm.WeightCompressor):
  """Defines how to apply the EPR algorithm."""

  def __init__(self, entropy_penalty):
    self.entropy_penalty = entropy_penalty

  def get_compressible_weights(self, original_layer):
    if isinstance(
        original_layer,
        (tf.keras.layers.Dense, tf.keras.layers.Conv1D, tf.keras.layers.Conv2D),
    ):
      if original_layer.use_bias:
        return [original_layer.kernel, original_layer.bias]
      else:
        return [original_layer.kernel]
    return []

  def init_training_weights(self, pretrained_weight: tf.Tensor):
    shape = pretrained_weight.shape
    dtype = pretrained_weight.dtype
    weight_name = "bias" if shape.rank == 1 else "kernel"

    if 1 <= shape.rank <= 2:
      # Bias or dense kernel.
      prior_shape = []
      self.add_training_weight(
          name=weight_name,
          shape=pretrained_weight.shape,
          dtype=pretrained_weight.dtype,
          initializer=tf.keras.initializers.Constant(pretrained_weight))
    elif 3 <= shape.rank <= 4:
      # Convolution kernel.
      kernel_shape = tf.shape(pretrained_weight)
      if shape.rank == 3:
        kernel_rdft = tf.signal.rfft(
            tf.transpose(pretrained_weight, (1, 2, 0)))
      else:
        kernel_rdft = tf.signal.rfft2d(
            tf.transpose(pretrained_weight, (2, 3, 0, 1)))
      kernel_rdft = tf.stack(
          [tf.math.real(kernel_rdft), tf.math.imag(kernel_rdft)], axis=-1)
      prior_shape = tf.shape(kernel_rdft)[2:]
      kernel_rdft /= tf.sqrt(tf.cast(tf.reduce_prod(kernel_shape[:-2]), dtype))
      self.add_training_weight(
          name="kernel_rdft",
          shape=kernel_rdft.shape,
          dtype=kernel_rdft.dtype,
          initializer=tf.keras.initializers.Constant(kernel_rdft))
      self.add_training_weight(
          name="kernel_shape",
          shape=kernel_shape.shape,
          dtype=kernel_shape.dtype,
          # TODO(jballe): If False, breaks optimize.create_layer_for_training().
          # If True, throws warnings that int tensors have no gradient.
          # trainable=False,
          initializer=tf.keras.initializers.Constant(kernel_shape))
    else:
      raise ValueError(
          f"Expected bias or kernel tensor with rank between 1 and 4, received "
          f"shape {self._shape}.")

    # Logarithm of quantization step size.
    log_step = tf.fill(prior_shape, tf.constant(-4, dtype=dtype))
    self.add_training_weight(
        name=f"{weight_name}_log_step",
        shape=log_step.shape,
        dtype=log_step.dtype,
        initializer=tf.keras.initializers.Constant(log_step))

    # Logarithm of scale of prior.
    log_scale = tf.fill(prior_shape, tf.constant(2.5, dtype=dtype))
    self.add_training_weight(
        name=f"{weight_name}_log_scale",
        shape=log_scale.shape,
        dtype=log_scale.dtype,
        initializer=tf.keras.initializers.Constant(log_scale))

  def project_training_weights(self, *training_weights) -> tf.Tensor:
    if len(training_weights) == 3:
      # Bias or dense kernel.
      weight, log_step, _ = training_weights
      step = tf.exp(log_step)
      return tfc.round_st(weight / step) * step
    else:
      # Convolution kernel.
      kernel_rdft, kernel_shape, log_step, _ = training_weights
      step = tf.exp(log_step)
      kernel_rdft = tfc.round_st(kernel_rdft / step)
      kernel_rdft *= step * tf.sqrt(
          tf.cast(tf.reduce_prod(kernel_shape[:-2]), kernel_rdft.dtype))
      kernel_rdft = tf.dtypes.complex(*tf.unstack(kernel_rdft, axis=-1))
      if kernel_rdft.shape.rank == 3:
        kernel = tf.signal.irfft(kernel_rdft, fft_length=kernel_shape[:-2])
        return tf.transpose(kernel, (2, 0, 1))
      else:
        kernel = tf.signal.irfft2d(kernel_rdft, fft_length=kernel_shape[:-2])
        return tf.transpose(kernel, (2, 3, 0, 1))

  def compress_training_weights(
      self, *training_weights: tf.Tensor) -> List[tf.Tensor]:
    if len(training_weights) == 3:
      # Bias or dense kernel.
      weight, log_step, log_scale = training_weights
      weight_shape = tf.shape(weight)
    else:
      # Convolution kernel.
      weight, weight_shape, log_step, log_scale = training_weights
    prior = tfc.NoisyLogistic(loc=0., scale=tf.exp(log_scale))
    em = tfc.ContinuousBatchedEntropyModel(
        prior, coding_rank=weight.shape.rank,
        compression=True, stateless=True, offset_heuristic=False)
    string = em.compress(weight / tf.exp(log_step))
    weight_shape = tf.cast(weight_shape, tf.uint16)
    return [string, weight_shape, log_step, em.cdf, em.cdf_offset]

  def decompress_weights(self, string, weight_shape, log_step,
                         cdf, cdf_offset) -> tf.Tensor:
    weight_shape = tf.cast(weight_shape, tf.int32)
    if weight_shape.shape[0] <= 2:
      # Bias or dense kernel.
      em = tfc.ContinuousBatchedEntropyModel(
          prior_shape=log_step.shape, cdf=cdf, cdf_offset=cdf_offset,
          coding_rank=weight_shape.shape[0], compression=True, stateless=True,
          offset_heuristic=False)
      return em.decompress(string, weight_shape) * tf.exp(log_step)
    else:
      # Convolution kernel.
      em = tfc.ContinuousBatchedEntropyModel(
          prior_shape=log_step.shape, cdf=cdf, cdf_offset=cdf_offset,
          coding_rank=weight_shape.shape[0] + 1, compression=True,
          stateless=True, offset_heuristic=False)
      kernel_rdft = em.decompress(string, weight_shape[-2:])
      kernel_rdft *= tf.exp(log_step) * tf.sqrt(
          tf.cast(tf.reduce_prod(weight_shape[:-2]), kernel_rdft.dtype))
      kernel_rdft = tf.dtypes.complex(*tf.unstack(kernel_rdft, axis=-1))
      if weight_shape.shape[0] == 3:
        kernel = tf.signal.irfft(kernel_rdft, fft_length=weight_shape[:-2])
        return tf.transpose(kernel, (2, 0, 1))
      else:
        kernel = tf.signal.irfft2d(kernel_rdft, fft_length=weight_shape[:-2])
        return tf.transpose(kernel, (2, 3, 0, 1))

  def compute_entropy(self, *training_weights) -> tf.Tensor:
    if len(training_weights) == 3:
      # Bias or dense kernel.
      weight, log_step, log_scale = training_weights
    else:
      # Convolution kernel.
      weight, _, log_step, log_scale = training_weights
    prior = tfc.NoisyLogistic(loc=0., scale=tf.exp(log_scale))
    em = tfc.ContinuousBatchedEntropyModel(
        prior, coding_rank=weight.shape.rank,
        compression=False, offset_heuristic=False)
    _, bits = em(weight / tf.exp(log_step), training=True)
    return bits

  def get_training_model(self, model: tf.keras.Model) -> tf.keras.Model:
    """Augments a model for training with EPR."""
    # pylint: disable=protected-access
    if (not isinstance(model, tf.keras.Sequential) and
        not model._is_graph_network):
      raise ValueError(
          "`compress_model` must be either a sequential or functional model.")
    # pylint: enable=protected-access

    entropies = []

    # Number of dimensions of original model weights. Used to bring
    # entropy_penalty into a more standardized range.
    weight_dims = tf.add_n([tf.size(w) for w in model.trainable_weights])

    def create_layer_for_training(layer):
      if not layer.built:
        raise ValueError(
            "Applying EPR currently requires passing in a built model.")
      train_layer = algorithm.create_layer_for_training(layer, algorithm=self)
      train_layer.build(layer.input_shape)
      for name in train_layer.attr_name_map.values():
        entropy = functools.partial(
            self.compute_entropy, *train_layer.training_weights[name])
        entropies.append(entropy)
      return train_layer

    def compute_entropy_loss():
      total_entropy = tf.add_n([e() for e in entropies])
      entropy_penalty = self.entropy_penalty / tf.cast(
          weight_dims, total_entropy.dtype)
      return total_entropy * entropy_penalty

    training_model = tf.keras.models.clone_model(
        model, clone_function=create_layer_for_training)
    training_model.add_loss(compute_entropy_loss)

    # TODO(jballe): It would be great to be able to track the entropy losses
    # combined during training. How to do this?
    # TODO(jballe): Some models might require training log_scale weights with a
    # different optimizer/learning rate. How to do this?
    return training_model

  def compress_model(self, model: tf.keras.Model) -> tf.keras.Model:
    """Compresses a model after training with EPR."""
    # pylint: disable=protected-access
    if (not isinstance(model, tf.keras.Sequential) and
        not model._is_graph_network):
      raise ValueError(
          "`compress_model` must be either a sequential or functional model.")
    # pylint: enable=protected-access

    def create_layer_for_inference(layer):
      return algorithm.create_layer_for_inference(layer, algorithm=self)

    return tf.keras.models.clone_model(
        model, clone_function=create_layer_for_inference)
