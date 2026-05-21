# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for default Quantizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow as tf

from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.quantization.keras.experimental.ternarization import ternarization_quantizers

TernarizationWeightsQuantizer = ternarization_quantizers.TernarizationWeightsQuantizer

keras = tf.keras


@keras_parameterized.run_all_keras_modes
class TernarizationWeightsQuantizerTest(tf.test.TestCase,
                                        parameterized.TestCase):

  @parameterized.parameters((keras.layers.Conv2D, {
      'filters': 5,
      'kernel_size': (2, 2)
  }))
  def testConstructsBetaStepVarsCorrectShape(self, layer_type, kwargs):
    quantizer = TernarizationWeightsQuantizer()

    model = keras.Sequential([layer_type(input_shape=(5, 2, 3), **kwargs)])
    layer = model.layers[0]

    training_vars = quantizer.build(layer.weights[0].shape, 'kernel', layer)
    # TODO(pulkitb): Add value test to ensure per-axis quantization is
    # happening properly. Probably to quant_ops_test.py
    quantizer(
        layer.weights[0],
        True,  # pylint: disable=unused-variable
        training_vars,
        layer=layer)

    beta = training_vars['beta']
    step = training_vars['step']
    self.assertEqual(5, beta.shape)
    self.assertEqual([], step.shape)


if __name__ == '__main__':
  tf.test.main()
