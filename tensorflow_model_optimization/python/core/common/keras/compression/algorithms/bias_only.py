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
"""BiasOnly algorithm, where the compress bias only."""
from typing import List

import tensorflow as tf

from tensorflow_model_optimization.python.core.common.keras.compression import algorithm
from tensorflow_model_optimization.python.core.keras.compat import keras


# TODO(tfmot): This algorithm is showcase for bias only compression. if we find
# better algorithm that can show better compressible weights coverage, then
# we can remove this algorithm.
class BiasOnly(algorithm.WeightCompressor):
  """Define how to apply BiasOnly algorithm."""

  # TODO(tfmot): communicate that `pretrained_weight` will sometimes
  # be a dummy tensor and sometimes be actual pretrained values during
  # its actual usage.
  def init_training_weights(
      self, pretrained_weight: tf.Tensor):
    bias_mean = tf.reduce_mean(pretrained_weight)
    bias_shape = tf.shape(pretrained_weight)

    # TODO(tfmot): note that it does not suffice to just have the initializer
    # to derive the shape from, in the case of a constant initializer.
    # The unit test fail without providing the shape.
    self.add_training_weight(
        name='bias_mean',
        shape=bias_mean.shape,
        dtype=bias_mean.dtype,
        initializer=keras.initializers.Constant(bias_mean),
    )
    self.add_training_weight(
        name='bias_shape',
        shape=bias_shape.shape,
        dtype=bias_shape.dtype,
        initializer=keras.initializers.Constant(bias_shape),
    )

  def decompress_weights(
      self, bias_mean: tf.Tensor, bias_shape: tf.Tensor) -> tf.Tensor:
    return tf.broadcast_to(bias_mean, bias_shape)

  def project_training_weights(
      self, bias_mean: tf.Tensor, bias_shape: tf.Tensor) -> tf.Tensor:
    return self.decompress_weights(bias_mean, bias_shape)

  def get_compressible_weights(
      self, original_layer: keras.layers.Layer
  ) -> List[str]:
    if isinstance(original_layer, keras.layers.Conv2D) or isinstance(
        original_layer, keras.layers.Dense
    ):
      return [original_layer.bias]
    return []

  def compress_model(self, to_optimize: keras.Model) -> keras.Model:
    """Model developer API for optimizing a model."""
    # pylint: disable=protected-access
    if (
        not isinstance(to_optimize, keras.Sequential)
        and not to_optimize._is_graph_network
    ):
      raise ValueError(
          '`compress_model` can only either be a keras Sequential or '
          'Functional model.'
      )
    # pylint: enable=protected-access

    def _optimize_layer(layer):
      # Require layer to be built so that the average of bias can be
      # initialized.
      if not layer.built:
        raise ValueError(
            'Applying BiasOnly currently requires passing in a built model')

      return algorithm.create_layer_for_training(layer, algorithm=self)

    return keras.models.clone_model(to_optimize, clone_function=_optimize_layer)
