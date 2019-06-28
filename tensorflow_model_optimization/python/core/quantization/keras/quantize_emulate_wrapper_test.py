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
"""Tests for QuantizeEmulateWrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.quantization.keras import quantize_emulate_wrapper

QuantizeEmulateWrapper = quantize_emulate_wrapper.QuantizeEmulateWrapper


class QuantizeEmulateWrapperTest(test.TestCase):

  def setUp(self):
    self.quant_params = {
        'num_bits': 8,
        'narrow_range': False,
        'symmetric': True
    }

  def testQuantizesWeightsInLayer(self):
    weights = lambda shape, dtype: np.array([[-1.0, 0.0], [0.0, 1.0]])
    model = keras.Sequential([
        QuantizeEmulateWrapper(
            keras.layers.Dense(2, kernel_initializer=weights),
            input_shape=(2,),
            **self.quant_params)
    ])

    # FakeQuant([-1.0, 1.0]) = [-0.9882355, 0.9882355]
    # Obtained from tf.fake_quant_with_min_max_vars
    self.assertAllClose(
        np.array([[-0.9882355, 0.9882355]]),
        # Inputs are all ones, so result comes directly from weights.
        model.predict(np.ones((1, 2))))


if __name__ == '__main__':
  test.main()
