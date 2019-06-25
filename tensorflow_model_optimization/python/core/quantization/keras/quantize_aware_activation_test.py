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
"""Tests for QuantizeAwareActivation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.platform import test

from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation

QuantizeAwareActivation = quantize_aware_activation.QuantizeAwareActivation


class QuantizeAwareQuantizationTest(test.TestCase):

  def testAppliesQuantizationPostActivation(self):
    model = keras.Sequential([
        QuantizeAwareActivation('relu', 'dense', num_bits=8)])

    x = np.array([-6.0, -3.0, 0.0, 0.05, 0.1, 3.0, 6.0])
    # All negative values are removed due to ReLU. The other expected values
    # are the border values of float buckets when [-6, 6] range is quantized to
    # 256 buckets.
    # Derived using `tf.fake_quant_with_min_max_vars`
    expected_activation = np.array(
        [0.0, 0.0, 0.0, 0.04705906, 0.09411764, 3.011765, 5.9764705]
    ).reshape(7, 1)

    self.assertAllClose(expected_activation, model.predict(x))

  def testAppliesQuantizationPreAndPostActivation(self):
    model = keras.Sequential([
        QuantizeAwareActivation('softmax', 'dense', num_bits=8)])

    x = np.array([[1.0, 2.0]])
    # expected_activation is determined using the float buckets when [-6, 6] is
    # quantized. Derived using `tf.fake_quant_with_min_max_vars`. For sigmoid,
    # quantization is applied twice.
    #
    # FakeQuant([1.0, 2.0]) = [0.9882355, 1.9764705]
    # Softmax([0.9882355, 1.9764705]) = [0.27126083, 0.72873914]
    # FakeQuant[0.27126083, 0.72873914]) = [0.28235292, 0.70588255]
    expected_activation = np.array([[0.28235292, 0.70588255]])

    self.assertAllClose(expected_activation, model.predict(x))

if __name__ == '__main__':
  test.main()
