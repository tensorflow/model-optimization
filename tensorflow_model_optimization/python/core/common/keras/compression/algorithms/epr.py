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

The "fast" version of EPR is inspired by the entropy code used in the paper:
> "Optimizing the Communication-Accuracy Trade-Off in Federated Learning with
> Rate-Distortion Theory"<br />
> N. Mitchell, J. Ballé, Z. Charles, J. Konečný<br />
> https://arxiv.org/abs/2201.02664
"""

import functools
from typing import Callable, List, Tuple
import tensorflow as tf
import tensorflow_compression as tfc
from tensorflow_model_optimization.python.core.common.keras.compression import algorithm
from tensorflow_model_optimization.python.core.keras.compat import keras


@tf.custom_gradient
def _to_complex_with_gradient(
    value: tf.Tensor) -> Tuple[tf.Tensor, Callable[[tf.Tensor], tf.Tensor]]:
  return tf.bitcast(value, tf.complex64), lambda g: tf.bitcast(g, tf.float32)


def _transform_dense_weight(
    weight: tf.Tensor,
    log_step: tf.Tensor,
    quantized: bool = True) -> tf.Tensor:
  """Transforms from latent to dense kernel or bias."""
  step = tf.exp(log_step)
  if not quantized:
    weight = tfc.round_st(weight / step)
  return weight * step


def _transform_conv_weight(
    kernel_rdft: tf.Tensor,
    kernel_shape: tf.Tensor,
    log_step: tf.Tensor,
    quantized: bool = True) -> tf.Tensor:
  """Transforms from latent to convolution kernel."""
  step = tf.exp(log_step)
  if not quantized:
    kernel_rdft = tfc.round_st(kernel_rdft / step)
  kernel_rdft *= step * tf.sqrt(
      tf.cast(tf.reduce_prod(kernel_shape[:-2]), kernel_rdft.dtype))
  kernel_rdft = _to_complex_with_gradient(kernel_rdft)
  if kernel_rdft.shape.rank == 3:
    # 1D convolution.
    kernel = tf.signal.irfft(kernel_rdft, fft_length=kernel_shape[:-2])
    return tf.transpose(kernel, (2, 0, 1))
  else:
    # 2D convolution.
    kernel = tf.signal.irfft2d(kernel_rdft, fft_length=kernel_shape[:-2])
    return tf.transpose(kernel, (2, 3, 0, 1))


class EPRBase(algorithm.WeightCompressor):
  """Defines how to apply the EPR algorithm."""

  _compressible_classes = (
      keras.layers.Dense,
      keras.layers.Conv1D,
      keras.layers.Conv2D,
  )

  def __init__(self, regularization_weight: float):
    super().__init__()
    self.regularization_weight = regularization_weight

  def get_compressible_weights(self, original_layer):
    if isinstance(original_layer, self._compressible_classes):
      if original_layer.use_bias:
        return [original_layer.kernel, original_layer.bias]
      else:
        return [original_layer.kernel]
    return []

  def _init_training_weights_reparam(
      self,
      pretrained_weight: tf.Tensor) -> Tuple[
          tf.TensorShape, tf.dtypes.DType, str]:
    """Initializes training weights needed for reparameterization."""
    shape = pretrained_weight.shape
    dtype = pretrained_weight.dtype
    weight_name = "bias" if shape.rank == 1 else "kernel"

    if 1 <= shape.rank <= 2:
      # Bias or dense kernel.
      self.add_training_weight(
          name=weight_name,
          shape=shape,
          dtype=dtype,
          initializer=keras.initializers.Constant(pretrained_weight),
      )
      prior_shape = tf.TensorShape(())
    elif 3 <= shape.rank <= 4:
      # Convolution kernel.
      kernel_shape = tf.shape(pretrained_weight)
      if shape.rank == 3:
        kernel_rdft = tf.signal.rfft(
            tf.transpose(pretrained_weight, (1, 2, 0)))
      else:
        kernel_rdft = tf.signal.rfft2d(
            tf.transpose(pretrained_weight, (2, 3, 0, 1)))
      kernel_rdft = tf.bitcast(kernel_rdft, tf.float32)
      kernel_rdft /= tf.sqrt(tf.cast(tf.reduce_prod(kernel_shape[:-2]), dtype))
      self.add_training_weight(
          name="kernel_rdft",
          shape=kernel_rdft.shape,
          dtype=kernel_rdft.dtype,
          initializer=keras.initializers.Constant(kernel_rdft),
      )
      self.add_training_weight(
          name="kernel_shape",
          shape=kernel_shape.shape,
          dtype=kernel_shape.dtype,
          # TODO(jballe): If False, breaks optimize.create_layer_for_training().
          # If True, throws warnings that int tensors have no gradient.
          # trainable=False,
          initializer=keras.initializers.Constant(kernel_shape),
      )
      prior_shape = kernel_rdft.shape[2:]
    else:
      raise ValueError(
          f"Expected bias or kernel tensor with rank between 1 and 4, received "
          f"shape {shape}.")

    # Logarithm of quantization step size.
    log_step = tf.fill(prior_shape, tf.constant(-4, dtype=dtype))
    self.add_training_weight(
        name=f"{weight_name}_log_step",
        shape=log_step.shape,
        dtype=log_step.dtype,
        initializer=keras.initializers.Constant(log_step),
    )

    return prior_shape, dtype, weight_name

  def get_training_model(self, model: keras.Model) -> keras.Model:
    """Augments a model for training with EPR."""
    if not (isinstance(model, keras.Sequential) or model._is_graph_network):  # pylint: disable=protected-access
      raise ValueError("`model` must be either sequential or functional.")

    training_model = keras.models.clone_model(
        model,
        clone_function=functools.partial(
            algorithm.create_layer_for_training, algorithm=self
        ),
    )
    training_model.build(model.input.shape)

    # Divide regularization weight by number of original model parameters to
    # bring it into a more standardized range.
    weight = self.regularization_weight / float(model.count_params())

    def regularization_loss(layer, name):
      return weight * self.regularization_loss(*layer.training_weights[name])

    for layer in training_model.layers:
      if not hasattr(layer, "attr_name_map"): continue
      for name in layer.attr_name_map.values():
        layer.add_loss(functools.partial(regularization_loss, layer, name))

    # TODO(jballe): It would be great to be able to track the entropy losses
    # combined during training. How to do this?
    # TODO(jballe): Some models might require training log_scale weights with a
    # different optimizer/learning rate. How to do this?
    return training_model

  def compress_model(self, model: keras.Model) -> keras.Model:
    """Compresses a model after training with EPR."""
    if not (isinstance(model, keras.Sequential) or model._is_graph_network):  # pylint: disable=protected-access
      raise ValueError("`model` must be either sequential or functional.")
    return keras.models.clone_model(
        model,
        clone_function=functools.partial(
            algorithm.create_layer_for_inference, algorithm=self
        ),
    )


class EPR(EPRBase):
  """Defines how to apply the EPR algorithm."""

  def init_training_weights(self, pretrained_weight: tf.Tensor):
    prior_shape, dtype, weight_name = self._init_training_weights_reparam(
        pretrained_weight)

    # In addition to reparameterization weights, this method also needs a
    # variable for the probability model (logarithm of scale of prior).
    log_scale = tf.fill(prior_shape, tf.constant(2.5, dtype=dtype))
    self.add_training_weight(
        name=f"{weight_name}_log_scale",
        shape=log_scale.shape,
        dtype=log_scale.dtype,
        initializer=keras.initializers.Constant(log_scale),
    )

  def project_training_weights(self, *training_weights: tf.Tensor) -> tf.Tensor:
    if len(training_weights) == 3:
      # Bias or dense kernel.
      return _transform_dense_weight(*training_weights[:-1], quantized=False)
    else:
      # Convolution kernel.
      return _transform_conv_weight(*training_weights[:-1], quantized=False)

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
    log_step = tf.cast(log_step, tf.float16)
    cdf = tf.cast(em.cdf, tf.int16)
    cdf_offset = tf.cast(em.cdf_offset, tf.int16)
    return [string, weight_shape, log_step, cdf, cdf_offset]

  def decompress_weights(
      self,
      string: tf.Tensor,
      weight_shape: tf.Tensor,
      log_step: tf.Tensor,
      cdf: tf.Tensor,
      cdf_offset: tf.Tensor) -> tf.Tensor:
    weight_shape = tf.cast(weight_shape, tf.int32)
    log_step = tf.cast(log_step, tf.float32)
    cdf = tf.cast(cdf, tf.int32)
    cdf_offset = tf.cast(cdf_offset, tf.int32)
    if weight_shape.shape[0] <= 2:
      # Bias or dense kernel.
      em = tfc.ContinuousBatchedEntropyModel(
          prior_shape=log_step.shape, cdf=cdf, cdf_offset=cdf_offset,
          coding_rank=weight_shape.shape[0], compression=True, stateless=True,
          offset_heuristic=False)
      weight = em.decompress(string, weight_shape)
      return _transform_dense_weight(weight, log_step, quantized=True)
    else:
      # Convolution kernel.
      em = tfc.ContinuousBatchedEntropyModel(
          prior_shape=log_step.shape, cdf=cdf, cdf_offset=cdf_offset,
          coding_rank=weight_shape.shape[0] + 1, compression=True,
          stateless=True, offset_heuristic=False)
      kernel_rdft = em.decompress(string, weight_shape[-2:])
      return _transform_conv_weight(
          kernel_rdft, weight_shape, log_step, quantized=True)

  def regularization_loss(self, *training_weights: tf.Tensor) -> tf.Tensor:
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


class FastEPR(EPRBase):
  """Defines how to apply a faster version of the EPR algorithm."""

  def __init__(self, regularization_weight: float, alpha: float = 1e-2):
    super().__init__(regularization_weight)
    self.alpha = alpha

  def init_training_weights(self, pretrained_weight: tf.Tensor):
    # The probability model is fixed, so only need reparameterization weights.
    self._init_training_weights_reparam(pretrained_weight)

  def project_training_weights(self, *training_weights: tf.Tensor) -> tf.Tensor:
    if len(training_weights) == 2:
      # Bias or dense kernel.
      return _transform_dense_weight(*training_weights, quantized=False)
    else:
      # Convolution kernel.
      return _transform_conv_weight(*training_weights, quantized=False)

  def compress_training_weights(
      self,
      *training_weights: tf.Tensor) -> List[tf.Tensor]:
    if len(training_weights) == 2:
      # Bias or dense kernel.
      weight, log_step = training_weights
      weight_shape = tf.shape(weight)
    else:
      # Convolution kernel.
      weight, weight_shape, log_step = training_weights
    em = tfc.PowerLawEntropyModel(
        coding_rank=weight.shape.rank, alpha=self.alpha)
    string = em.compress(weight / tf.exp(log_step))
    weight_shape = tf.cast(weight_shape, tf.uint16)
    log_step = tf.cast(log_step, tf.float16)
    return [string, weight_shape, log_step]

  def decompress_weights(
      self,
      string: tf.Tensor,
      weight_shape: tf.Tensor,
      log_step: tf.Tensor) -> tf.Tensor:
    weight_shape = tf.cast(weight_shape, tf.int32)
    log_step = tf.cast(log_step, tf.float32)
    if weight_shape.shape[0] <= 2:
      # Bias or dense kernel.
      em = tfc.PowerLawEntropyModel(
          coding_rank=weight_shape.shape[0], alpha=self.alpha)
      weight = em.decompress(string, weight_shape)
      return _transform_dense_weight(weight, log_step, quantized=True)
    else:
      # Convolution kernel.
      em = tfc.PowerLawEntropyModel(
          coding_rank=weight_shape.shape[0] + 1, alpha=self.alpha)
      kernel_rdft = em.decompress(
          string, tf.concat([weight_shape[-2:], tf.shape(log_step)], 0))
      return _transform_conv_weight(
          kernel_rdft, weight_shape, log_step, quantized=True)

  def regularization_loss(self, *training_weights: tf.Tensor) -> tf.Tensor:
    if len(training_weights) == 2:
      # Bias or dense kernel.
      weight, log_step = training_weights
    else:
      # Convolution kernel.
      weight, _, log_step = training_weights
    em = tfc.PowerLawEntropyModel(
        coding_rank=weight.shape.rank, alpha=self.alpha)
    return em.penalty(weight / tf.exp(log_step))
