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
"""SVD algorithm, where the training and inference graphs are the same."""
from typing import List

import tensorflow as tf

from tensorflow_model_optimization.python.core.common.keras.compression import algorithm
from tensorflow_model_optimization.python.core.keras.compat import keras


class SVD(algorithm.WeightCompressor):
  """Define how to apply SVD algorithm.

  This periodic update and scheduling base SVD algorithm update the original
  weights to make lower rank by SVD for each update_freq steps. During the
  warmup steps, It adjust the rank from the original to target rank gradually.
  """

  def __init__(self, rank, update_freq=100, warmup_step=1000):
    self.rank = rank
    self.update_freq = update_freq
    self.warmup_step = warmup_step

  # TODO(tfmot): communicate that `pretrained_weight` will sometimes
  # be a dummy tensor and sometimes be actual pretrained values during
  # its actual usage.
  def init_training_weights(
      self, pretrained_weight: tf.Tensor):
    self.add_training_weight(
        name='w',
        shape=pretrained_weight.shape,
        dtype=pretrained_weight.dtype,
        initializer=keras.initializers.Constant(pretrained_weight),
    )
    self.add_training_weight(
        name='step',
        shape=(),
        dtype=tf.int32,
        initializer=keras.initializers.Constant(0),
    )

  def decompress_weights(self, u: tf.Tensor, sv: tf.Tensor) -> tf.Tensor:
    return tf.matmul(u, sv)

  def project_training_weights(
      self, weight: tf.Tensor, step: tf.Tensor) -> tf.Tensor:
    weight_rank = tf.math.minimum(weight.shape[-1], weight.shape[-2])
    self.update_training_weight(step, step + 1)
    if step % self.update_freq == 0:
      rank = self.rank
      if step < self.warmup_step:
        rank = tf.cast(tf.math.round(
            weight_rank * (self.warmup_step - step)
            + self.rank * step
            ) / self.warmup_step, tf.int32)
      rank = tf.math.minimum(rank, weight_rank)

      s, u, v = tf.linalg.svd(weight)

      if len(weight.shape) == 2:
        # FC Layer
        s = s[:rank]
        u = u[:, :rank]
        v = v[:, :rank]
      elif len(weight.shape) == 4:
        # Conv2D Layer
        s = s[:, :, :rank]
        u = u[:, :, :, :rank]
        v = v[:, :, :, :rank]
      else:
        raise NotImplementedError('Only for dimension=2 or 4 is supported.')

      sv = tf.matmul(tf.linalg.diag(s), v, adjoint_b=True)

      new_weight = tf.matmul(u, sv)
      self.update_training_weight(weight, new_weight)

    return weight

  def compress_training_weights(self, weight: tf.Tensor, _) -> List[tf.Tensor]:
    rank = self.rank
    s, u, v = tf.linalg.svd(weight)

    if len(weight.shape) == 2:
      # FC Layer
      s = s[:rank]
      u = u[:, :rank]
      v = v[:, :rank]
    elif len(weight.shape) == 4:
      # Conv2D Layer
      s = s[:, :, :rank]
      u = u[:, :, :, :rank]
      v = v[:, :, :, :rank]
    else:
      raise NotImplementedError('Only for dimension=2 or 4 is supported.')

    sv = tf.matmul(tf.linalg.diag(s), v, adjoint_b=True)

    return [u, sv]

  def get_compressible_weights(
      self, original_layer: keras.layers.Layer
  ) -> List[str]:
    if isinstance(original_layer, (keras.layers.Conv2D, keras.layers.Dense)):
      return [original_layer.kernel]
    return []

  def optimize_model(self, to_optimize: keras.Model) -> keras.Model:
    """Model developer API for optimizing a model for training.

    The returned model should be used for compression aware training.
    Args:
      to_optimize: The model to be optimize.
    Returns:
      A wrapped model that has compression optimizers.
    """
    # pylint: disable=protected-access
    if (
        not isinstance(to_optimize, keras.Sequential)
        and not to_optimize._is_graph_network
    ):
      raise ValueError(
          '`optimize_model` can only either be a keras Sequential or '
          'Functional model.'
      )
    # pylint: enable=protected-access

    def _optimize_layer(layer):
      # Require layer to be built so that the SVD-factorized weights
      # can be initialized from the weights.
      if not layer.built:
        raise ValueError(
            'Applying SVD currently requires passing in a built model')

      return algorithm.create_layer_for_training(layer, algorithm=self)

    return keras.models.clone_model(to_optimize, clone_function=_optimize_layer)

  def compress_model(self, to_compress: keras.Model) -> keras.Model:
    """Model developer API for optimizing a model for inference.

    Args:
      to_compress: The model that trained for compression. This model should
        generated from the `optimize_model` method.
    Returns:
      A compressed model for the inference.
    """
    def _optimize_layer(layer):
      # Require layer to be built so that the SVD-factorized weights
      # can be initialized from the weights.
      if not layer.built:
        raise ValueError(
            'Applying SVD currently requires passing in a built model')

      return algorithm.create_layer_for_inference(layer, algorithm=self)

    return keras.models.clone_model(to_compress, clone_function=_optimize_layer)
