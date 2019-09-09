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
"""Functional test which fully trains quantized models and verifies accuracy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl.testing import parameterized

from tensorflow.python import keras
from tensorflow.python.platform import test

from tensorflow_model_optimization.python.core.keras.testing import test_utils_mnist
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import utils as test_utils

quantize_annotate = quantize.quantize_annotate
quantize_apply = quantize.quantize_apply


class QuantizeFunctionalTest(test.TestCase, parameterized.TestCase):

  # TODO(pulkitb): Parameterize test and include functional mnist, and
  # other RNN models.
  def testQuantizesMnist(self):
    model = test_utils_mnist.sequential_model()
    x_train, y_train, x_test, y_test = test_utils_mnist.preprocessed_data()

    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=500)
    _, model_accuracy = model.evaluate(x_test, y_test, verbose=0)

    quantized_model = quantize_apply(quantize_annotate(model))
    quantized_model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    quantized_model.fit(x_train, y_train, batch_size=500)
    _, quantized_model_accuracy = model.evaluate(x_test, y_test, verbose=0)

    self.assertGreater(quantized_model_accuracy, 0.6)

    _, quantized_keras_file = tempfile.mkstemp('.h5')
    _, quantized_tflite_file = tempfile.mkstemp('.tflite')

    keras.models.save_model(quantized_model, quantized_keras_file)
    with quantize.quantize_scope():
      test_utils.convert_keras_to_tflite(
          model_path=quantized_keras_file,
          output_path=quantized_tflite_file,
          is_quantized=True)
    quantized_model_tflite_accuracy = test_utils_mnist.eval_tflite(
        quantized_tflite_file, is_quantized=True)

    # TODO(pulkitb): Check if Forward Pass of TF and TF-Lite are the same.

    # Ensure accuracy for quantized TF and TFLite models are similar to original
    # model. There is no clear way to measure quantization, but for MNIST
    # results which differ a lot likely suggest an error in quantization.
    self.assertAllClose(
        model_accuracy, quantized_model_accuracy,
        rtol=0.2, atol=0.2)
    self.assertAllClose(
        quantized_model_accuracy, quantized_model_tflite_accuracy,
        rtol=0.2, atol=0.2)


if __name__ == '__main__':
  test.main()
