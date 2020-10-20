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
"""Public APIs for algorithm developer using weight compression API."""
import abc
from typing import List, Any
import dataclasses

import tensorflow as tf

from tensorflow_model_optimization.python.core.common.keras.compression.internal import optimize


@dataclasses.dataclass
class WeightRepr:
  """Dataclass that wraps `tf.keras.layers.Layer.add_weight` parameters."""
  name: Any = None
  shape: Any = None
  dtype: Any = None
  initializer: Any = None
  regularizer: Any = None
  trainable: Any = None
  constraint: Any = None
  partitioner: Any = None
  use_resource: Any = None
  synchronization: Any = tf.VariableSynchronization.AUTO
  aggregation: Any = tf.compat.v1.VariableAggregation.NONE


class WeightCompressionAlgorithm(metaclass=abc.ABCMeta):
  """Interface for weight compression algorithm that acts on a per-layer basis.

     This allows both options of either decompressing during inference or
     decompressing prior to inference (where compression occurs by applying a
     tool such as zip to the model file).

     This interface is a purely functional one.
  """

  @abc.abstractmethod
  def init_training_weights_repr(
      self, pretrained_weight: tf.Tensor) -> List[WeightRepr]:
    """Create training weight representations for initializing layer variables.

    Args:
      pretrained_weight: tf.Tensor of a pretrained weight of a layer that will
        be compressed eventually.

    Returns:
      A list of `WeightRepr`, a container for arguments to
      `tf.keras.layers.Layer.add_weight`for each tf.Variable to create.
    """

  def compress(self, training_weights: List[tf.Tensor]) -> List[tf.Tensor]:
    """Define the operations to compress a single weight after training.

    'Compress' can refer to making the weight more amenable to compression
    or actually compress the weight.

    The default is an identity.

    Args:
      training_weights: tf.Tensors representing all variables used during
        training, for a single compressible weight, in the order returned in
        `init_training_weights_repr`.

    Returns:
      List of tf.Tensors to set to compressed or more compressible form.
    """
    return training_weights

  def decompress(self, compressed_weights: List[tf.Tensor]) -> tf.Tensor:
    """Define the operations to decompress a single weight’s compressed form during inference.

    The default is an identity. TODO(): actually isn't.

    Args:
       compressed_weights: tf.Tensors representing a single weight’s compressed
         form, coming from what’s returned in `compress`.

    Returns:
      A tf.Tensor representing the decompressed `compressed_weights`.
    """
    return compressed_weights[0]

  @abc.abstractmethod
  def training(self, training_weights: List[tf.Tensor]) -> tf.Tensor:
    """Define a piece of the forward pass during training, which operates on a single compressible weight.

    TODO(tfmot): throw this error.
    The default throws an error when training occurs.

    Args:
       training_weights: tf.Tensors representing any variables used during
         training, for a single compressible weight, in the order returned in
         `init_training_weights_repr`.

    Returns:
       tf.Tensor to set the compressible weight to.
    """


def create_layer_for_training(
    layer: tf.keras.layers.Layer,
    algorithm: WeightCompressionAlgorithm) -> tf.keras.layers.Layer:
  return optimize.create_layer_for_training(layer, algorithm)


def create_layer_for_inference(
    layer_for_training: tf.keras.layers.Layer,
    algorithm: WeightCompressionAlgorithm) -> tf.keras.layers.Layer:
  return optimize.create_layer_for_inference(layer_for_training, algorithm)
