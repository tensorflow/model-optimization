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
"""Quantize Annotate Wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.platform import test

from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate
from tensorflow_model_optimization.python.core.quantization.keras import quantize_emulatable_layer

QuantizeEmulatableLayer = quantize_emulatable_layer.QuantizeEmulatableLayer


class QuantizeAnnotateTest(test.TestCase):

  def setUp(self):
    self.quant_params = {
        'num_bits': 8,
        'narrow_range': True,
        'symmetric': True
    }

  def testRaisesErrorForUnsupportedLayer(self):
    class CustomLayer(keras.layers.Dense):
      pass

    with self.assertRaises(ValueError):
      quantize_annotate.QuantizeAnnotate(CustomLayer(10), **self.quant_params)

  def testAnnotatesCustomQuantizableLayer(self):
    class CustomLayerQuantizable(keras.layers.Dense, QuantizeEmulatableLayer):
      def get_quantizable_weights(self):  # pylint: disable=g-wrong-blank-lines
        return [self.kernel]

      def set_quantizable_weights(self, weights):
        self.kernel = weights[0]

    annotated_layer = quantize_annotate.QuantizeAnnotate(
        CustomLayerQuantizable(10), **self.quant_params)

    self.assertIsInstance(annotated_layer.layer, CustomLayerQuantizable)
    self.assertEqual(
        self.quant_params, annotated_layer.get_quantize_params())

  def testAnnotatesKerasLayer(self):
    layer = keras.layers.Dense(5, activation='relu', input_shape=(10,))
    model = keras.Sequential([layer])

    annotated_model = keras.Sequential([
        quantize_annotate.QuantizeAnnotate(
            layer, input_shape=(10,), **self.quant_params)])

    annotated_layer = annotated_model.layers[0]
    self.assertIsInstance(annotated_layer.layer, keras.layers.Dense)
    self.assertEqual(
        self.quant_params, annotated_layer.get_quantize_params())

    # Annotated model should not affect computation. Returns same results.
    x_test = np.random.rand(10, 10)
    self.assertAllEqual(model.predict(x_test), annotated_model.predict(x_test))

if __name__ == '__main__':
  test.main()
