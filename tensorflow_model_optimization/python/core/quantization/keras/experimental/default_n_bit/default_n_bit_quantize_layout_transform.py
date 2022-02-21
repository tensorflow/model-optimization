# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Default 8-bit layout transformation for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import default_n_bit_transforms
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer


keras = tf.keras


class DefaultNBitQuantizeLayoutTransform(
    quantize_layout_transform.QuantizeLayoutTransform):
  """Default model transformations."""

  def __init__(self, num_bits_weight: int = 8, num_bits_activation: int = 8):
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

  def apply(self, model, layer_quantize_map):
    """Implement default 8-bit transforms.

    Currently this means the following.
      1. Pull activations into layers, and apply fuse activations. (TODO)
      2. Modify range in incoming layers for Concat. (TODO)
      3. Fuse Conv2D/DepthwiseConv2D + BN into single layer.

    Args:
      model: Keras model to be quantized.
      layer_quantize_map: Map with keys as layer names, and values as dicts
        containing custom `QuantizeConfig`s which may have been passed with
        layers.

    Returns:
      (Transformed Keras model to better match TensorFlow Lite backend, updated
      layer quantize map.)
    """

    transforms = [
        default_n_bit_transforms.InputLayerQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.SeparableConv1DQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.SeparableConvQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.Conv2DReshapeBatchNormReLUQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.Conv2DReshapeBatchNormActivationQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.Conv2DBatchNormReLUQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.Conv2DBatchNormActivationQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.Conv2DReshapeBatchNormQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.Conv2DBatchNormQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.ConcatTransform6Inputs(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.ConcatTransform5Inputs(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.ConcatTransform4Inputs(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.ConcatTransform3Inputs(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.ConcatTransform(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.LayerReLUQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.LayerReluActivationQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.DenseBatchNormQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.DenseBatchNormReLUQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.DenseBatchNormActivationQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
    ]
    return model_transformer.ModelTransformer(
        model, transforms,
        set(layer_quantize_map.keys()), layer_quantize_map).transform()
