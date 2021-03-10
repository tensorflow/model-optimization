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


class SVD(algorithm.WeightCompressor):
  """Define how to apply SVD algorithm."""

  def __init__(self, rank):
    self.rank = rank

  # TODO(tfmot): communicate that `pretrained_weight` will sometimes
  # be a dummy tensor and sometimes be actual pretrained values during
  # its actual usage.
  def init_training_weights(self, pretrained_weight: tf.Tensor):
    rank = self.rank
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
    self.add_training_weight(
        name='u',
        shape=u.shape,
        dtype=u.dtype,
        initializer=tf.keras.initializers.Constant(u))
    self.add_training_weight(
        name='sv',
        shape=sv.shape,
        dtype=sv.dtype,
        initializer=tf.keras.initializers.Constant(sv))

  def decompress_weights(self, u: tf.Tensor, sv: tf.Tensor) -> tf.Tensor:
    return tf.matmul(u, sv)

  def project_training_weights(self, u: tf.Tensor, sv: tf.Tensor) -> tf.Tensor:
    return self.decompress_weights(u, sv)

  def get_compressible_weights(
      self, original_layer: tf.keras.layers.Layer) -> List[str]:
    if isinstance(original_layer, tf.keras.layers.Conv2D) or \
       isinstance(original_layer, tf.keras.layers.Dense):
      return [original_layer.kernel]
    return []

  def compress_model(self, to_optimize: tf.keras.Model) -> tf.keras.Model:
    """Model developer API for optimizing a model."""
    # pylint: disable=protected-access
    if not isinstance(to_optimize, tf.keras.Sequential) \
        and not to_optimize._is_graph_network:
      raise ValueError(
          '`compress_model` can only either be a tf.keras Sequential or '
          'Functional model.')
    # pylint: enable=protected-access

    def _optimize_layer(layer):
      # Require layer to be built so that the SVD-factorized weights
      # can be initialized from the weights.
      if not layer.built:
        raise ValueError(
            'Applying SVD currently requires passing in a built model')

      return algorithm.create_layer_for_training(layer, algorithm=self)

    return tf.keras.models.clone_model(
        to_optimize, clone_function=_optimize_layer)
