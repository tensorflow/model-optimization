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
"""Tests for compress API functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_model_optimization.python.core.internal.compression.keras import compress

keras = tf.keras
layers = keras.layers

tf.enable_v2_behavior()


class SimpleLayerWiseConfig(compress.LayerVariableWiseCompressionConfig):

  def init(self, kernel):
    kernel_shape = tf.shape(kernel)
    return [kernel_shape, tf.math.reduce_mean(kernel, keepdims=True)]

  def decompress(self, kernel_shape, compressed_kernel):
    return [tf.broadcast_to(compressed_kernel, kernel_shape)]


class LayerVariableWiseCompressionConfigTest(tf.test.TestCase):

  def testSimpleLayerWiseConfig(self):
    input_layer = tf.keras.layers.Input(shape=(28, 28, 3), name='input')
    x = layers.Conv2D(8, [3, 3])(input_layer)
    x = layers.GlobalAveragePooling2D()(x)
    output_layer = layers.Dense(10)(x)
    model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
    model.build((1, 32, 32, 3))

    layer_name_to_weight_keys_map = {}
    for layer in model.layers:
      if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
        layer_name_to_weight_keys_map[layer.name] = ['kernel']
    params = compress.LayerVariableWiseParameters(layer_name_to_weight_keys_map)

    config = SimpleLayerWiseConfig.build_for_model(model, params)

    model_training = compress.convert_from_model(
        model,
        config,
        phase=compress.CompressionModelPhase.training)

    model_compressed = compress.convert_to_compressed_phase_from_training_phase(
        model_training,
        config)

    self.assertEqual(model.count_params(), 314)
    self.assertEqual(model_training.count_params(), 26)
    self.assertEqual(model_compressed.count_params(), 26)

# TODO(kimjaehong): Add more tests for classes in compress.py.

if __name__ == '__main__':
  tf.test.main()
