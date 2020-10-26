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


class SVDParams(object):
  """Define container for parameters for SVD algorithm."""

  def __init__(self, rank):
    self.rank = rank


class SVD(algorithm.WeightCompressionAlgorithm):
  """Define how to apply SVD algorithm."""

  def __init__(self, params):
    self.params = params

  # TODO(tfmot): communicate that `pretrained_weight` will sometimes
  # be a dummy tensor and sometimes be actual pretrained values during
  # its actual usage.
  def init_training_weights_repr(
      self, pretrained_weight: tf.Tensor) -> List[algorithm.WeightRepr]:
    rank = self.params.rank
    s, u, v = tf.linalg.svd(pretrained_weight)

    if len(pretrained_weight.shape) == 2:
      # FC Layer
      s = s[:rank]
      u = u[:, :rank]
      v = v[:, :rank]
    elif len(pretrained_weight.shape) == 4:
      # Conv2D Layer
      s = s[:, :, :rank]
      u = u[:, :, :, :rank]
      v = v[:, :, :, :rank]
    else:
      raise NotImplementedError('Only for dimension=2 or 4 is supported.')

    sv = tf.matmul(tf.linalg.diag(s), v, adjoint_b=True)

    # TODO(tfmot): note that it does not suffice to just have the initializer
    # to derive the shape from, in the case of a constant initializer.
    # The unit test fail without providing the shape.
    return [
        algorithm.WeightRepr(
            name='u',
            shape=u.shape,
            initializer=tf.keras.initializers.Constant(u)),
        algorithm.WeightRepr(
            name='sv',
            shape=sv.shape,
            initializer=tf.keras.initializers.Constant(sv))
    ]

  def decompress(self, u: tf.Tensor, sv: tf.Tensor) -> tf.Tensor:
    return tf.matmul(u, sv)

  def training(self, u: tf.Tensor, sv: tf.Tensor) -> tf.Tensor:
    return self.decompress(u, sv)


def optimize(to_optimize: tf.keras.Model, params: SVDParams) -> tf.keras.Model:
  """Model developer API for optimizing a model."""

  def _optimize_layer(layer):
    # Require layer to be built so that the SVD-factorized weights
    # can be initialized from the weights.
    if not layer.built:
      raise ValueError(
          'Applying SVD currently requires passing in a built model')

    return algorithm.create_layer_for_training(layer, algorithm=SVD(params))

  return tf.keras.models.clone_model(
      to_optimize, clone_function=_optimize_layer)
