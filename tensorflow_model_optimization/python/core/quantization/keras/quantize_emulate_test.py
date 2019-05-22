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
"""Tests for keras pruning wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import keras
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.quantization.keras.quantize_emulate import QuantizeEmulate
from tensorflow_model_optimization.python.core.quantization.keras.quantize_emulate_wrapper import QuantizeEmulateWrapper


class QuantizeEmulateTest(test.TestCase):

  def setUp(self):
    self.conv_layer = keras.layers.Conv2D(32, 4, input_shape=(28, 28, 1))
    self.dense_layer = keras.layers.Dense(10)
    self.params = {'num_bits': 8}

  def _assert_quant_model(self, model_layers):
    self.assertIsInstance(model_layers[0], QuantizeEmulateWrapper)
    self.assertIsInstance(model_layers[1], QuantizeEmulateWrapper)

    self.assertEqual(model_layers[0].layer, self.conv_layer)
    self.assertEqual(model_layers[1].layer, self.dense_layer)

  def testQuantizeEmulateSequential(self):
    model = keras.models.Sequential([
        self.conv_layer,
        self.dense_layer
    ])

    quant_model = QuantizeEmulate(model, **self.params)

    self._assert_quant_model(quant_model.layers)

  def testQuantizeEmulateList(self):
    quant_layers = QuantizeEmulate([self.conv_layer, self.dense_layer],
                                   **self.params)

    self._assert_quant_model(quant_layers)


if __name__ == '__main__':
  test.main()
