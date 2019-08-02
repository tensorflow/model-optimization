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
# pylint: disable=protected-access
"""Quantization specific utilities for generating, saving, testing, and evaluating models."""

import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow_model_optimization.python.core.quantization.keras.quantize_emulate_wrapper import QuantizeEmulateWrapper


def convert_mnist_to_tflite(model_path, output_path, custom_objects=None):
  """Convert Keras mnist model to TFLite."""
  if custom_objects is None:
    custom_objects = {}
  custom_objects.update({'QuantizeEmulateWrapper': QuantizeEmulateWrapper})

  converter = tf.lite.TFLiteConverter.from_keras_model_file(
      model_path,
      custom_objects=custom_objects)

  converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
  input_arrays = converter.get_input_arrays()
  converter.quantized_input_stats = {
      input_arrays[0]: (0., 255.)
  }  # mean, std_dev

  tflite_model = converter.convert()
  open(output_path, 'wb').write(tflite_model)


def _get_fake_quant_values(layer):
  min_weights = []
  max_weights = []
  for _, min_weight, max_weight in layer._weight_vars:
    min_weights.append(K.get_value(min_weight))
    max_weights.append(K.get_value(max_weight))
  min_activation = K.get_value(layer._min_activation)
  max_activation = K.get_value(layer._max_activation)

  return min_weights, max_weights, min_activation, max_activation


def assert_fake_quant_equivalence(test_case, model1, model2):
  """Validate that FakeQuant operators in the QuantizeEmulateWrapper of two models are equal.

  Args:
    test_case: test.TestCase instance.
    model1: first model for comparison.
    model2: second model for comparison.
  """
  # This ensures that the number of layers and QuantizeEmulateWrapped layers
  # is equal.
  test_case.assertEqual(model1.get_config(), model2.get_config())
  for l in xrange(len(model1.layers)):
    l1 = model1.layers[l]
    l2 = model2.layers[l]
    if isinstance(l1, QuantizeEmulateWrapper):
      min_weights_1, max_weights_1, min_activation_1, max_activation_1 = _get_fake_quant_values(
          l1)
      min_weights_2, max_weights_2, min_activation_2, max_activation_2 = _get_fake_quant_values(
          l2)

      test_case.assertListEqual(min_weights_1, min_weights_2)
      test_case.assertListEqual(max_weights_1, max_weights_2)

      test_case.assertEqual(min_activation_1, min_activation_2)
      test_case.assertEqual(max_activation_1, max_activation_2)
