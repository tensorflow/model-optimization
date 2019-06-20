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
"""Quantize Annotate Wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.platform import test

from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate


class QuantizeAnnotateTest(test.TestCase):

  def testAppliesWrapperToAllClasses(self):
    layer = keras.layers.Dense(5, activation='relu', input_shape=(10,))

    model = keras.Sequential([layer])
    wrapped_model = keras.Sequential([
        quantize_annotate.QuantizeAnnotate(
            layer, num_bits=8, input_shape=(10,))])

    x_test = np.random.rand(10, 10)
    self.assertAllEqual(model.predict(x_test), wrapped_model.predict(x_test))

if __name__ == '__main__':
  test.main()
