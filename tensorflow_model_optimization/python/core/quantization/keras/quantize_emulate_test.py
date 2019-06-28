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
"""Tests for quantize API functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate as quant_annotate
from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras import quantize_emulate
from tensorflow_model_optimization.python.core.quantization.keras import quantize_emulate_wrapper

quantize_annotate = quantize_emulate.quantize_annotate
QuantizeEmulate = quantize_emulate.QuantizeEmulate
QuantizeEmulateWrapper = quantize_emulate_wrapper.QuantizeEmulateWrapper
QuantizeAwareActivation = quantize_aware_activation.QuantizeAwareActivation


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


class QuantizeApplyTest(test.TestCase):

  def setUp(self):
    self.quant_params1 = {
        'num_bits': 8,
        'narrow_range': True,
        'symmetric': True
    }
    self.quant_params2 = {
        'num_bits': 4,
        'narrow_range': False,
        'symmetric': False
    }

  # Validation tests

  def testRaisesErrorIfNotKerasModel(self):
    with self.assertRaises(ValueError):
      quantize_emulate.quantize_apply(keras.layers.Dense(32))

  def testRaisesErrorIfKerasSubclassedModel(self):
    class MyModel(keras.Model):
      def call(self, inputs, training=None, mask=None):  # pylint: disable=g-wrong-blank-lines
        return inputs

    with self.assertRaises(ValueError):
      quantize_emulate.quantize_apply(MyModel())

  def testRaisesErrorNoAnnotatedLayers_Sequential(self):
    model = keras.Sequential([
        keras.layers.Dense(10), keras.layers.Dropout(0.4)])

    with self.assertRaises(ValueError):
      quantize_emulate.quantize_apply(model)

  def testRaisesErrorNoAnnotatedLayers_Functional(self):
    inputs = keras.Input(shape=(10,))
    x = keras.layers.Dense(32, activation='relu')(inputs)
    results = keras.layers.Dense(5, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=results)

    with self.assertRaises(ValueError):
      quantize_emulate.quantize_apply(model)

  # Quantization Apply Tests

  def _get_annotated_sequential_model(self):
    return keras.Sequential([
        quantize_annotate(keras.layers.Conv2D(32, 5), input_shape=(28, 28, 1),
                          **self.quant_params1),
        quantize_annotate(keras.layers.Dense(10), **self.quant_params2)
    ])

  def _get_annotated_functional_model(self):
    inputs = keras.Input(shape=(28, 28, 1))
    x = quantize_annotate(
        keras.layers.Conv2D(32, 5), **self.quant_params1)(inputs)
    results = quantize_annotate(keras.layers.Dense(10), **self.quant_params2)(x)

    return keras.Model(inputs=inputs, outputs=results)

  def _assert_weights_equal_value(self, annotated_weights, emulated_weights):
    annotated_weight_values = K.batch_get_value(annotated_weights)
    emulated_weight_values = K.batch_get_value(emulated_weights)

    self.assertEqual(len(annotated_weight_values), len(emulated_weight_values))
    for aw, ew in zip(annotated_weight_values, emulated_weight_values):
      self.assertAllClose(aw, ew)

  def _assert_weights_different_objects(
      self, annotated_weights, emulated_weights):
    self.assertEqual(len(annotated_weights), len(emulated_weights))
    for aw, ew in zip(annotated_weights, emulated_weights):
      self.assertNotEqual(id(aw), id(ew))

  def _assert_layer_emulated(
      self, annotated_layer, emulated_layer, exclude_keys=None):
    self.assertIsInstance(emulated_layer, QuantizeEmulateWrapper)

    self.assertEqual(annotated_layer.get_quantize_params(),
                     emulated_layer.get_quantize_params())

    # Extract configs of the inner layers they wrap.
    annotated_config = annotated_layer.layer.get_config()
    emulated_config = emulated_layer.layer.get_config()

    # The underlying layers aren't always exactly the same. For example,
    # activations in the underlying layers might be replaced. Exclude keys
    # if required.
    if exclude_keys:
      for key in exclude_keys:
        annotated_config.pop(key)
        emulated_config.pop(key)

    self.assertEqual(annotated_config, emulated_config)

    def _sort_weights(weights):
      # Variables are named `quantize_annotate0/kernel:0` and
      # `quantize_emulate0/kernel:0`. Strip layer name to sort.
      return sorted(weights, key=lambda w: w.name.split('/')[1])

    annotated_weights = _sort_weights(annotated_layer.trainable_weights)
    emulated_weights = _sort_weights(emulated_layer.trainable_weights)

    # Quantized model should pick the same weight values from the original
    # model. However, they should not be the same weight objects. We don't
    # want training the quantized model to change weights in the original model.
    self._assert_weights_different_objects(annotated_weights, emulated_weights)
    self._assert_weights_equal_value(annotated_weights, emulated_weights)

  def _assert_model_emulated(
      self, annotated_model, emulated_model, exclude_keys=None):
    for annotated_layer, emulated_layer in zip(annotated_model.layers,
                                               emulated_model.layers):
      if isinstance(emulated_layer, keras.layers.InputLayer):
        continue

      self._assert_layer_emulated(annotated_layer, emulated_layer, exclude_keys)

  def testAppliesQuantizationToAnnotatedModel_Sequential(self):
    model = self._get_annotated_sequential_model()

    quantized_model = quantize_emulate.quantize_apply(model)

    self._assert_model_emulated(model, quantized_model)

  def testAppliesQuantizationToAnnotatedModel_Functional(self):
    model = self._get_annotated_functional_model()

    quantized_model = quantize_emulate.quantize_apply(model)

    self._assert_model_emulated(model, quantized_model)

  # Transformation Tests

  def testQuantizesActivationsWithinLayer_Sequential(self):
    quant_params = {'num_bits': 8, 'symmetric': True}
    model = keras.Sequential([
        quantize_annotate(
            keras.layers.Conv2D(32, 5, activation='relu'),
            input_shape=(28, 28, 1),
            **quant_params)
    ])

    quantized_model = quantize_emulate.quantize_apply(model)

    # We expect activation to be modified.
    self._assert_model_emulated(model, quantized_model, ['activation'])

    conv_layer = quantized_model.layers[0].layer
    self.assertIsInstance(conv_layer.activation, QuantizeAwareActivation)
    self.assertEqual(
        keras.activations.get('relu'), conv_layer.activation.activation)
    self.assertEqual(keras.layers.Conv2D, conv_layer.activation.parent_layer)
    self.assertEqual(quant_params, conv_layer.activation.get_quantize_params())

  def testQuantizesActivationsWithinLayer_Functional(self):
    quant_params = {'num_bits': 8, 'symmetric': True}

    inputs = keras.Input(shape=(28, 28, 1))
    results = quantize_annotate(
        keras.layers.Conv2D(32, 5, activation='relu'),
        **self.quant_params1)(inputs)
    model = keras.Model(inputs=inputs, outputs=results)

    quantized_model = quantize_emulate.quantize_apply(model)

    # We expect activation to be modified.
    self._assert_model_emulated(model, quantized_model, ['activation'])

    conv_layer = quantized_model.layers[1].layer
    self.assertIsInstance(conv_layer.activation, QuantizeAwareActivation)
    self.assertEqual(
        keras.activations.get('relu'), conv_layer.activation.activation)
    self.assertEqual(keras.layers.Conv2D, conv_layer.activation.parent_layer)
    self.assertEqual(quant_params, conv_layer.activation.get_quantize_params())


if __name__ == '__main__':
  test.main()
