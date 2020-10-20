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
"""SVD algorithm, where the training and inference graphs are different."""
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
    return [
        algorithm.WeightRepr(
            name='w',
            shape=pretrained_weight.shape,
            initializer=tf.keras.initializers.Constant(pretrained_weight))
    ]

  def decompress(self, u: tf.Tensor, sv: tf.Tensor) -> tf.Tensor:
    return tf.matmul(u, sv)

  def compress(self, training_weights: List[tf.Tensor]) -> List[tf.Tensor]:
    assert len(training_weights) == 1
    weight = training_weights[0]

    rank = self.params.rank
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

  # TODO(tfmot): remove in this example, which is just post-training.
  def training(self, training_weights: List[tf.Tensor]) -> tf.Tensor:
    return training_weights[0]


# TODO(tfmot): consider if we can simplify `create_model_for_training` and
# `create_model_for_inference` into a single API for algorithm developers.
def optimize(to_optimize: tf.keras.Model, params: SVDParams) -> tf.keras.Model:
  """Model developer API for optimizing a model."""

  def _create_layer_for_training(layer):
    # Require layer to be built so that the SVD-factorized weights
    # can be initialized from the weights.
    if not layer.built:
      raise ValueError(
          'Applying SVD currently requires passing in a built model')

    return algorithm.create_layer_for_training(layer, algorithm=SVD(params))

  def _create_layer_for_inference(layer):
    return algorithm.create_layer_for_inference(layer, algorithm=SVD(params))

  intermediate_model = tf.keras.models.clone_model(
      to_optimize, clone_function=_create_layer_for_training)

  return tf.keras.models.clone_model(
      intermediate_model, clone_function=_create_layer_for_inference)
