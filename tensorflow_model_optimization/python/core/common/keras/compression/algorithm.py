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
  args: Any = None
  kwargs: Any = None


class WeightCompressor(metaclass=abc.ABCMeta):
  """Interface for weight compression algorithm that acts on a per-layer basis.

     This allows both options of either decompressing during inference or
     decompressing prior to inference (where compression occurs by applying a
     tool such as zip to the model file).

     This interface is a purely functional one.
  """
  update_ops = []  # type: List

  # TODO(tfmot): Consider separate from algorithm API for custom layer supports.
  def get_compressible_weights(
      self, original_layer: tf.keras.layers.Layer) -> List[tf.Variable]:
    """Define compressible weights for each layer.

    Args:
       original_layer: tf.keras.layers.Layer representing a layer from the
       original model.

    Returns:
       List of compressible weights for the given layer.
    """
    del original_layer
    return []

  @abc.abstractmethod
  def init_training_weights(
      self, pretrained_weight: tf.Tensor):
    """Initialize training weights for the compressible weight.

    It calls the `add_training_weight` to add a training weight for a given
    `pretrained_weight`. A `pretrained_weight` can have multiple training
    weights. We initialize the training weights for each compressible
    weight by just calling this function for each.

    Args:
      pretrained_weight: tf.Tensor of a pretrained weight of a layer that will
        be compressed eventually.
    """

  def add_training_weight(
      self, *args, **kwargs):
    """Add a training weight for the compressible weight.

    When this method is called from the `init_training_weights`, this adds
    training weights for the pretrained_weight that is the input of the
    `init_training_weights`.

    Args:
      *args: Passed through to training_model.add_weight.
      **kwargs: Passed through to training_model.add_weight.
    """
    weight_repr = WeightRepr(args=args, kwargs=kwargs)
    if hasattr(self, 'weight_reprs'):
      self.weight_reprs.append(weight_repr)
    else:
      self.weight_reprs = [weight_repr]

  @abc.abstractmethod
  def project_training_weights(
      self, *training_weights: tf.Tensor) -> tf.Tensor:
    """Define a piece of the forward pass during training.

    It operates on a single compressible weight.
    The default throws an error when training occurs.

    Args:
       *training_weights: tf.Tensors representing any variables used during
         training, for a single compressible weight, in the order returned in
         `init_training_weights`.

    Returns:
       tf.Tensor to set the compressible weight to.
    """

  def init_update_ops(self, tensor_weight_pairs):
    self.update_ops = []
    self.tensor_weight_pairs = tensor_weight_pairs

  def update_training_weight(
      self, training_weight: tf.Tensor, value: tf.Tensor):
    """Add training weight assign op to the model update list.

     This method is for the case that training weight should update to a
    specific value not from the model optimizer. It will throw an error if it
    can't find the training weight.

     This method should called in project_training_weights. During the training,
    We collect all update_training_weight calls and make an UpdateOp for each
    call. Finally, we put all these update ops to model.add_update.

    Args:
      training_weight: tf.Tensor representing a training weight.
      value: tf.Tensor representing a value to be assigned to the training
        weight.
    Raises:
      ValueError if it can't find the training weight.
    """
    for tensor, weight in self.tensor_weight_pairs:
      if training_weight is tensor:
        self.update_ops.append(weight.assign(value))
        return

    raise ValueError('Training weight not found. Please call '
                     'the update_training_weight with given training '
                     'weight tensor.')

  def get_update_ops(self):
    return self.update_ops

  def compress_training_weights(
      self, *training_weights: tf.Tensor) -> List[tf.Tensor]:
    """Define the operations to compress a single weight’s training form.

    'compress_training_weights' can refer to making the weight more amenable to
    compression or actually compress the weight.

    The default is an identity.

    Args:
      *training_weights: tf.Tensors representing all variables used during
        training, for a single compressible weight, in the order returned in
        `init_training_weights`.

    Returns:
      List of tf.Tensors to set to compressed or more compressible form.
    """
    return list(training_weights)

  @abc.abstractmethod
  def decompress_weights(
      self, *compressed_weights: tf.Tensor) -> tf.Tensor:
    """Define the operations to decompress a single weight’s compressed form.

    The default is an identity.

    Args:
       *compressed_weights: tf.Tensors representing a single weight’s compressed
         form, coming from what’s returned in `compress`.

    Returns:
      A tf.Tensor representing the decompressed `compressed_weights`.
    """


def create_layer_for_training(
    layer: tf.keras.layers.Layer,
    algorithm: WeightCompressor) -> tf.keras.layers.Layer:
  return optimize.create_layer_for_training(layer, algorithm)


def create_layer_for_inference(
    layer_for_training: tf.keras.layers.Layer,
    algorithm: WeightCompressor) -> tf.keras.layers.Layer:
  return optimize.create_layer_for_inference(layer_for_training, algorithm)
