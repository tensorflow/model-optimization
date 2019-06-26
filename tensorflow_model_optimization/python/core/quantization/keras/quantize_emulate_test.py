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

import numpy as np

from tensorflow.python import keras
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate as quant_annotate
from tensorflow_model_optimization.python.core.quantization.keras import quantize_emulate
from tensorflow_model_optimization.python.core.quantization.keras.quantize_emulate import QuantizeEmulate
from tensorflow_model_optimization.python.core.quantization.keras.quantize_emulate_wrapper import QuantizeEmulateWrapper

quantize_annotate = quantize_emulate.quantize_annotate


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


class QuantizeAnnotateTest(test.TestCase):

  def setUp(self):
    self.quant_params = {
        'num_bits': 8,
        'narrow_range': True,
        'symmetric': True
    }

  def _assertQuantParams(self, layer, quant_params):
    layer_params = {
        'num_bits': layer._num_bits,
        'narrow_range': layer._narrow_range,
        'symmetric': layer._symmetric
    }
    self.assertEqual(quant_params, layer_params)

  def _assertWrappedLayer(self, layer, quant_params):
    self.assertIsInstance(layer, quant_annotate.QuantizeAnnotate)
    self._assertQuantParams(layer, quant_params)

  def _assertWrappedSequential(self, model, quant_params):
    for layer in model.layers:
      self._assertWrappedLayer(layer, quant_params)

  def testQuantizeAnnotateLayer(self):
    layer = keras.layers.Dense(10, input_shape=(5,))
    wrapped_layer = quantize_annotate(
        layer, input_shape=(5,), **self.quant_params)

    self._assertWrappedLayer(wrapped_layer, self.quant_params)

    inputs = np.random.rand(1, 5)
    model = keras.Sequential([layer])
    wrapped_model = keras.Sequential([wrapped_layer])

    # Both models should have the same results, since quantize_annotate does
    # not modify behavior.
    self.assertAllEqual(model.predict(inputs), wrapped_model.predict(inputs))

  def testQuantizeAnnotateModel(self):
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(5,)),
        keras.layers.Dropout(0.4)
    ])
    annotated_model = quantize_annotate(model, **self.quant_params)

    self._assertWrappedSequential(annotated_model, self.quant_params)

    inputs = np.random.rand(1, 5)
    self.assertAllEqual(model.predict(inputs), annotated_model.predict(inputs))

  def testQuantizeAnnotateModel_HasAnnotatedLayers(self):
    layer_params = {'num_bits': 4, 'narrow_range': False, 'symmetric': False}
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(5,)),
        quantize_annotate(keras.layers.Dense(5), **layer_params)
    ])

    annotated_model = quantize_annotate(model, **self.quant_params)

    self._assertWrappedLayer(annotated_model.layers[0], self.quant_params)
    self._assertWrappedLayer(annotated_model.layers[1], layer_params)
    # Ensure an already annotated layer is not wrapped again.
    self.assertIsInstance(annotated_model.layers[1].layer, keras.layers.Dense)

    inputs = np.random.rand(1, 5)
    self.assertAllEqual(model.predict(inputs), annotated_model.predict(inputs))


if __name__ == '__main__':
  test.main()
