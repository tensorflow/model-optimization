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
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.platform import test

from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras import quantizers

QuantizeAwareActivation = quantize_aware_activation.QuantizeAwareActivation
MovingAverageQuantizer = quantizers.MovingAverageQuantizer


class QuantizeAwareQuantizationTest(test.TestCase):

  def setUp(self):
    super(QuantizeAwareQuantizationTest, self).setUp()
    self.quantizer = MovingAverageQuantizer(
        num_bits=8, per_axis=False, symmetric=True)

  class TestLayer(keras.layers.Layer):

    def call(self, inputs, training=None):
      if training is None:
        training = K.learning_phase()

      self.activation.training = training
      return self.activation(inputs)

    def compute_output_shape(self, input_shape):
      return input_shape

  def testRaisesError_NotKerasBuiltinActivation(self):
    layer = self.TestLayer()

    def custom_quantize(x):
      return x

    with self.assertRaises(ValueError) as cm:
      QuantizeAwareActivation(custom_quantize, self.quantizer, 0, layer)
    self.assertEqual(
        str(cm.exception), QuantizeAwareActivation._CUSTOM_ACTIVATION_ERR_MSG)

  def testAppliesQuantizationPostActivation(self):
    layer = self.TestLayer()
    layer.activation = QuantizeAwareActivation(
        activations.get('relu'), self.quantizer, 0, layer)

    model = keras.Sequential([layer])

    x = np.array([-6.0, -3.0, 0.0, 0.05, 0.1, 3.0, 6.0])
    # All negative values are removed due to ReLU. The other expected values
    # are the border values of float buckets when [-6, 6] range is quantized to
    # 256 buckets.
    # Derived using `tf.fake_quant_with_min_max_vars`
    expected_activation = np.array(
        [0.0, 0.0, 0.0, 0.04705906, 0.09411764, 3.011765,
         5.9764705]).reshape(7, 1)

    self.assertAllClose(expected_activation, model.predict(x))

  def testAppliesQuantizationPreAndPostActivation(self):
    layer = self.TestLayer()
    layer.activation = QuantizeAwareActivation(
        activations.get('softmax'), self.quantizer, 0, layer)

    model = keras.Sequential([layer])

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

  def testSerializationReturnsWrappedActivation_BuiltInActivation(self):
    activation = activations.get('softmax')
    quantize_activation = QuantizeAwareActivation(
        activation, self.quantizer, 0, self.TestLayer())

    expected_config = {
        'class_name': 'QuantizeAwareActivation',
        'config': {'activation': 'softmax'}
    }
    serialized_quantize_activation = serialize_keras_object(quantize_activation)

    self.assertEqual(expected_config, serialized_quantize_activation)

    deserialized_activation = deserialize_keras_object(
        serialized_quantize_activation,
        custom_objects={'QuantizeAwareActivation': QuantizeAwareActivation})

    self.assertEqual(activation, deserialized_activation)


if __name__ == '__main__':
  test.main()
