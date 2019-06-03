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
"""End to End tests for the Quantization API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import numpy as np

from tensorflow.python import keras
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.keras import test_utils
from tensorflow_model_optimization.python.core.quantization.keras.quantize_emulate import QuantizeEmulate
from tensorflow_model_optimization.python.core.quantization.keras.quantize_emulate_wrapper import QuantizeEmulateWrapper
from tensorflow_model_optimization.python.core.quantization.keras.utils import assert_fake_quant_equivalence


class QuantizeEmulateIntegrationTest(test.TestCase):

  def setUp(self):
    super(QuantizeEmulateIntegrationTest, self).setUp()
    self.params = {'num_bits': 8}

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

  def _check_models_match(self, model1, model2):
    self.assertEqual(model1.input.get_shape().as_list(),
                     model2.input.get_shape().as_list())
    self.assertEqual(model1.get_config(), model2.get_config())

    assert_fake_quant_equivalence(self, model1, model2)

    # Check predictions match.
    input_data = np.random.randn(
        *self._batch(model1.input.get_shape().as_list(), 1))
    model1_result = model1.predict(input_data)
    model2_result = model2.predict(input_data)
    np.testing.assert_almost_equal(model1_result, model2_result)

  # TODO(alanchiao): add similar test for mnist.
  def testQuantizeSaveAndRestore(self):
    model = QuantizeEmulate(test_utils.build_simple_dense_model(),
                            **self.params)
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # Verify serialization correctness persists after training.
    model.fit(
        np.random.rand(20, 10),
        keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        batch_size=20,
    )

    _, keras_file = tempfile.mkstemp('.h5')
    keras.models.save_model(model, keras_file)
    loaded_model = keras.models.load_model(
        keras_file,
        custom_objects={'QuantizeEmulateWrapper': QuantizeEmulateWrapper})
    self._check_models_match(model, loaded_model)


if __name__ == '__main__':
  test.main()
