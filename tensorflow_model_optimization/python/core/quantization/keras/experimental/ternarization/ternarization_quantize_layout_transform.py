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
"""Default ternarization layout transformation for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.experimental.ternarization import ternarization_transforms
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer

keras = tf.keras


class TernarizationQuantizeLayoutTransform(
    quantize_layout_transform.QuantizeLayoutTransform):
  """Default model transformations."""

  def apply(self, model, layer_quantize_map):
    """Implement default ternarization transforms.

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
        ternarization_transforms.InputLayerQuantize(),
        ternarization_transforms.SeparableConv1DQuantize(),
        ternarization_transforms.SeparableConvQuantize(),
        ternarization_transforms.Conv2DReshapeBatchNormReLUQuantize(),
        ternarization_transforms.Conv2DReshapeBatchNormActivationQuantize(),
        ternarization_transforms.Conv2DBatchNormReLUQuantize(),
        ternarization_transforms.Conv2DBatchNormActivationQuantize(),
        ternarization_transforms.Conv2DReshapeBatchNormQuantize(),
        ternarization_transforms.Conv2DBatchNormQuantize(),
        ternarization_transforms.ConcatTransform6Inputs(),
        ternarization_transforms.ConcatTransform5Inputs(),
        ternarization_transforms.ConcatTransform4Inputs(),
        ternarization_transforms.ConcatTransform3Inputs(),
        ternarization_transforms.ConcatTransform(),
        ternarization_transforms.DenseBatchNormQuantize(),
        ternarization_transforms.DenseBatchNormReLUQuantize(),
        ternarization_transforms.DenseBatchNormActivationQuantize(),
        ternarization_transforms.LayerReLUQuantize(),
        ternarization_transforms.LayerReluActivationQuantize(),
    ]
    return model_transformer.ModelTransformer(model, transforms,
                                              set(layer_quantize_map.keys()),
                                              layer_quantize_map).transform()
