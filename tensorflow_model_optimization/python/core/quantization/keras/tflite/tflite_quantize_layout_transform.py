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
"""Layout Transformation for quantization in TFLite.

Module: tfmot.quantization.keras.tflite
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.tflite import tflite_transforms

keras = tf.keras


class TFLiteQuantizeLayoutTransform(
    quantize_layout_transform.QuantizeLayoutTransform):
  """Model transformations for TFLite."""

  def apply(self, model, layer_quantize_map):
    """Implement TFLite transforms.

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

    # TODO(pulkitb): Sequential models not supported yet. Remove once support is
    # added.
    if isinstance(model, keras.Sequential):
      return model, layer_quantize_map

    transforms = [
        tflite_transforms.InputLayerQuantize(),
        tflite_transforms.Conv2DBatchNormReLUQuantize(),
        tflite_transforms.Conv2DBatchNormActivationQuantize(),
        tflite_transforms.Conv2DBatchNormQuantize(),
    ]

    return model_transformer.ModelTransformer(
        model, transforms,
        layer_quantize_map.keys(), layer_quantize_map).transform()
