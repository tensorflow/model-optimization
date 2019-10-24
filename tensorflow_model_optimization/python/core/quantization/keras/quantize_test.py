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
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.platform import test

from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate as quantize_annotate_mod
from tensorflow_model_optimization.python.core.quantization.keras import quantize_provider as quantize_provider_mod
from tensorflow_model_optimization.python.core.quantization.keras import quantize_wrapper as quantize_wrapper_mod
from tensorflow_model_optimization.python.core.quantization.keras.tflite import tflite_quantize_registry

quantize_annotate = quantize.quantize_annotate
quantize_apply = quantize.quantize_apply
QuantizeAnnotate = quantize_annotate_mod.QuantizeAnnotate
QuantizeWrapper = quantize_wrapper_mod.QuantizeWrapper


class _TestQuantizeProvider(quantize_provider_mod.QuantizeProvider):

  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    pass

  def get_config(self):
    return {}


class QuantizeAnnotateTest(test.TestCase):

  def _assertWrappedLayer(self, layer, quantize_provider=None):
    self.assertIsInstance(layer, quantize_annotate_mod.QuantizeAnnotate)
    self.assertEqual(quantize_provider, layer.quantize_provider)

  def _assertWrappedModel(self, model):
    for layer in model.layers:
      self._assertWrappedLayer(layer)

  def testQuantizeAnnotateLayer(self):
    layer = keras.layers.Dense(10, input_shape=(5,))
    wrapped_layer = quantize_annotate(layer, input_shape=(5,))

    self._assertWrappedLayer(wrapped_layer)

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
    annotated_model = quantize_annotate(model)

    self._assertWrappedModel(annotated_model)

    inputs = np.random.rand(1, 5)
    self.assertAllEqual(model.predict(inputs), annotated_model.predict(inputs))

  def testQuantizeAnnotateModel_HasAnnotatedLayers(self):
    quantize_provider = _TestQuantizeProvider()

    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(5,)),
        quantize_annotate_mod.QuantizeAnnotate(
            keras.layers.Dense(5), quantize_provider=quantize_provider)
    ])
    annotated_model = quantize_annotate(model)

    self._assertWrappedLayer(annotated_model.layers[0])
    self._assertWrappedLayer(annotated_model.layers[1], quantize_provider)
    # Ensure an already annotated layer is not wrapped again.
    self.assertIsInstance(annotated_model.layers[1].layer, keras.layers.Dense)

    inputs = np.random.rand(1, 5)
    self.assertAllEqual(model.predict(inputs), annotated_model.predict(inputs))


class QuantizeApplyTest(test.TestCase):

  # Validation tests

  def testRaisesErrorIfNotKerasModel(self):
    with self.assertRaises(ValueError):
      quantize_apply(keras.layers.Dense(32))

  def testRaisesErrorIfKerasSubclassedModel(self):
    class MyModel(keras.Model):
      def call(self, inputs, training=None, mask=None):  # pylint: disable=g-wrong-blank-lines
        return inputs

    with self.assertRaises(ValueError):
      quantize_apply(MyModel())

  def testRaisesErrorNoAnnotatedLayers_Sequential(self):
    model = keras.Sequential([
        keras.layers.Dense(10), keras.layers.Dropout(0.4)])

    with self.assertRaises(ValueError):
      quantize_apply(model)

  def testRaisesErrorNoAnnotatedLayers_Functional(self):
    inputs = keras.Input(shape=(10,))
    x = keras.layers.Dense(32, activation='relu')(inputs)
    results = keras.layers.Dense(5, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=results)

    with self.assertRaises(ValueError):
      quantize_apply(model)

  def testRaisesErrorModelNotBuilt(self):
    model = keras.Sequential([
        quantize_annotate(keras.layers.Dense(10))])

    self.assertFalse(model.built)
    with self.assertRaises(ValueError):
      quantize_apply(model)

  # Helper functions to verify quantize wrapper applied correctly.

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

  def _assert_layer_quantized(
      self, annotate_wrapper, quantize_wrapper, exclude_keys=None):
    self.assertIsInstance(quantize_wrapper, QuantizeWrapper)

    # Extract configs of the inner layers they wrap.
    annotated_config = annotate_wrapper.layer.get_config()
    quantized_config = quantize_wrapper.layer.get_config()

    # The underlying layers aren't always exactly the same. For example,
    # activations in the underlying layers might be replaced. Exclude keys
    # if required.
    if exclude_keys:
      for key in exclude_keys:
        annotated_config.pop(key)
        quantized_config.pop(key)

    self.assertEqual(annotated_config, quantized_config)

    def _sort_weights(weights):
      # Variables are named `quantize_annotate0/kernel:0` and
      # `quantize_emulate0/kernel:0`. Strip layer name to sort.
      return sorted(weights, key=lambda w: w.name.split('/')[1])

    annotated_weights = _sort_weights(annotate_wrapper.trainable_weights)
    quantized_weights = _sort_weights(quantize_wrapper.trainable_weights)

    # Quantized model should pick the same weight values from the original
    # model. However, they should not be the same weight objects. We don't
    # want training the quantized model to change weights in the original model.
    self._assert_weights_different_objects(annotated_weights, quantized_weights)
    self._assert_weights_equal_value(annotated_weights, quantized_weights)

  def _assert_model_quantized(
      self, annotated_model, quantized_model, exclude_keys=None):
    for layer_annotated, layer_quantized in \
        zip(annotated_model.layers, quantized_model.layers):

      if not isinstance(layer_annotated, QuantizeAnnotate):
        self.assertNotIsInstance(layer_quantized, QuantizeWrapper)
        continue

      self._assert_layer_quantized(
          layer_annotated, layer_quantized, exclude_keys)

  # quantize_apply Tests

  class CustomLayer(keras.layers.Dense):
    pass

  def testQuantize_RaisesErrorIfNoQuantizeProvider(self):
    annotated_model = keras.Sequential([
        QuantizeAnnotate(self.CustomLayer(3), input_shape=(2,))])

    with generic_utils.custom_object_scope({'CustomLayer': self.CustomLayer}):
      with self.assertRaises(RuntimeError):
        quantize_apply(annotated_model)

  def testQuantize_UsesBuiltinQuantizeProvider(self):
    annotated_model = keras.Sequential([
        QuantizeAnnotate(keras.layers.Dense(3), input_shape=(2,))])

    quantized_model = quantize_apply(annotated_model)
    quantized_layer = quantized_model.layers[0]

    # 'activation' gets replaced while quantizing the model. Hence excluded
    # from equality checks.
    self._assert_layer_quantized(
        annotated_model.layers[0], quantized_layer, ['activation'])
    self.assertIsInstance(quantized_layer.quantize_provider,
                          tflite_quantize_registry.TFLiteQuantizeProvider)

  def testQuantize_UsesQuantizeProviderFromUser_NoBuiltIn(self):
    annotated_model = keras.Sequential([
        QuantizeAnnotate(self.CustomLayer(3), input_shape=(2,),
                         quantize_provider=_TestQuantizeProvider())])

    with generic_utils.custom_object_scope(
        {'CustomLayer': self.CustomLayer,
         '_TestQuantizeProvider': _TestQuantizeProvider}):
      quantized_model = quantize_apply(annotated_model)
    quantized_layer = quantized_model.layers[0]

    self._assert_layer_quantized(annotated_model.layers[0], quantized_layer)
    self.assertIsInstance(
        quantized_layer.quantize_provider, _TestQuantizeProvider)

  def testQuantize_PreferenceToUserSpecifiedQuantizeProvider(self):
    annotated_model = keras.Sequential([
        QuantizeAnnotate(keras.layers.Dense(3), input_shape=(2,),
                         quantize_provider=_TestQuantizeProvider())])

    with generic_utils.custom_object_scope(
        {'_TestQuantizeProvider': _TestQuantizeProvider}):
      quantized_model = quantize_apply(annotated_model)
    quantized_layer = quantized_model.layers[0]

    self._assert_layer_quantized(annotated_model.layers[0], quantized_layer)
    self.assertIsInstance(
        quantized_layer.quantize_provider, _TestQuantizeProvider)

  def testAppliesQuantizationToAnnotatedModel_Sequential(self):
    model = keras.Sequential([
        keras.layers.Conv2D(32, 5, input_shape=(28, 28, 1), activation='relu'),
        QuantizeAnnotate(keras.layers.Dense(10, activation='relu')),
        QuantizeAnnotate(keras.layers.Dense(5, activation='softmax')),
    ])

    quantized_model = quantize_apply(model)

    self._assert_model_quantized(model, quantized_model, ['activation'])

  def testAppliesQuantizationToAnnotatedModel_Functional(self):
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, 5, activation='relu')(inputs)
    x = QuantizeAnnotate(keras.layers.Dense(10, activation='relu'))(x)
    results = QuantizeAnnotate(keras.layers.Dense(5, activation='softmax'))(x)
    model = keras.Model(inputs=inputs, outputs=results)

    quantized_model = quantize_apply(model)

    self._assert_model_quantized(model, quantized_model, ['activation'])


if __name__ == '__main__':
  test.main()
