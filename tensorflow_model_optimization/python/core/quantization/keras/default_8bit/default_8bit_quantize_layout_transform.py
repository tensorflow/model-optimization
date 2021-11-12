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
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_transforms
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer

keras = tf.keras


class Default8BitQuantizeLayoutTransform(
    quantize_layout_transform.QuantizeLayoutTransform):
  """Default model transformations."""

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
        default_8bit_transforms.InputLayerQuantize(),
        default_8bit_transforms.SeparableConv1DQuantize(),
        default_8bit_transforms.SeparableConvQuantize(),
        default_8bit_transforms.Conv2DReshapeBatchNormReLUQuantize(),
        default_8bit_transforms.Conv2DReshapeBatchNormActivationQuantize(),
        default_8bit_transforms.Conv2DBatchNormReLUQuantize(),
        default_8bit_transforms.Conv2DBatchNormActivationQuantize(),
        default_8bit_transforms.Conv2DReshapeBatchNormQuantize(),
        default_8bit_transforms.Conv2DBatchNormQuantize(),
        default_8bit_transforms.ConcatTransform6Inputs(),
        default_8bit_transforms.ConcatTransform5Inputs(),
        default_8bit_transforms.ConcatTransform4Inputs(),
        default_8bit_transforms.ConcatTransform3Inputs(),
        default_8bit_transforms.ConcatTransform(),
        default_8bit_transforms.DenseBatchNormQuantize(),
        default_8bit_transforms.DenseBatchNormReLUQuantize(),
        default_8bit_transforms.DenseBatchNormActivationQuantize(),
        default_8bit_transforms.LayerReLUQuantize(),
        default_8bit_transforms.LayerReluActivationQuantize(),
    ]
    return model_transformer.ModelTransformer(
        model, transforms,
        set(layer_quantize_map.keys()), layer_quantize_map).transform()
