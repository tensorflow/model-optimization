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
"""Integration test which ensures user facing code paths work."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.platform import test

from tensorflow_model_optimization.python.core.keras import test_utils
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import utils


class QuantizeIntegrationTest(test.TestCase, parameterized.TestCase):

  @staticmethod
  def _batch(dims, batch_size):
    """Adds provided batch_size to existing dims.

    If dims is (None, 5, 2), returns (batch_size, 5, 2)

    Args:
      dims: Dimensions
      batch_size: batch_size

    Returns:
      dims with batch_size added as first parameter of list.
    """
    if dims[0] is None:
      dims[0] = batch_size
    return dims

  def _assert_models_equal(self, model1, model2):
    self.assertEqual(model1.get_config(), model2.get_config())
    self.assertAllClose(model1.get_weights(), model2.get_weights())

    inputs = np.random.randn(
        *self._batch(model1.input.get_shape().as_list(), 1))
    self.assertAllClose(model1.predict(inputs), model2.predict(inputs))

  # TODO(pulkitb): Parameterize and add more model/runtime options.
  def testSerialization(self):
    model = test_utils.build_simple_dense_model()

    quantized_model = quantize.quantize(model)
    quantized_model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    quantized_model.fit(
        np.random.rand(20, 10),
        tf.keras.utils.to_categorical(
            np.random.randint(5, size=(20, 1)), 5),
        batch_size=20)

    _, model_file = tempfile.mkstemp('.h5')
    keras.models.save_model(quantized_model, model_file)
    with quantize.quantize_scope():
      loaded_model = keras.models.load_model(model_file)

    self._assert_models_equal(quantized_model, loaded_model)


if __name__ == '__main__':
  test.main()
