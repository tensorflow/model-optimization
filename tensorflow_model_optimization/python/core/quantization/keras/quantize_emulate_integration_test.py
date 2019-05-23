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

from tensorflow.python import keras
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.keras import test_utils
from tensorflow_model_optimization.python.core.quantization.keras.quantize_emulate import QuantizeEmulate


class QuantizeEmulateIntegrationTest(test.TestCase):

  def setUp(self):
    super(QuantizeEmulateIntegrationTest, self).setUp()
    self.params = {'num_bits': 8}

  # TODO(alanchiao): test restore and that save and restore
  # models are equivalent.
  def testQuantizeSave(self):
    model = QuantizeEmulate(test_utils.build_simple_dense_model(),
                            **self.params)
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    _, keras_file = tempfile.mkstemp('.h5')
    keras.models.save_model(model, keras_file)


if __name__ == '__main__':
  test.main()
