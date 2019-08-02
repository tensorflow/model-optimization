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

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.quantization.keras import quantize_emulate_wrapper

QuantizeEmulateWrapper = quantize_emulate_wrapper.QuantizeEmulateWrapper


class QuantizeEmulateWrapperTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(QuantizeEmulateWrapperTest, self).setUp()
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

  @parameterized.parameters(
      (layers.convolutional.Conv1D, (3, 6), {
          'filters': 4,
          'kernel_size': 2
      }),
      (layers.convolutional.Conv2D, (4, 6, 1), {
          'filters': 4,
          'kernel_size': (2, 2)
      }),
      (layers.convolutional.Conv2DTranspose, (7, 6, 3), {
          'filters': 2,
          'kernel_size': (3, 3)
      }),
      (layers.convolutional.Conv3D, (5, 7, 6, 3), {
          'filters': 2,
          'kernel_size': (3, 3, 3)
      }),
      (layers.convolutional.Conv3DTranspose, (5, 7, 6, 3), {
          'filters': 2,
          'kernel_size': (3, 3, 3)
      }),
      # TODO(pulkitb): Add missing SeparableConv layers. The initializers are
      # different, so will need a change.
      (layers.core.Dense, (6,), {
          'units': 4
      }),
      (layers.local.LocallyConnected1D, (3, 6), {
          'filters': 4,
          'kernel_size': 2
      }),
      (layers.local.LocallyConnected2D, (4, 6, 1), {
          'filters': 4,
          'kernel_size': (2, 2)
      }),
  )
  def testQuantizesWeights_KerasLayers(self, layer, input_shape, kwargs):
    self.weights = None

    def _get_random_weights(shape, dtype):  # pylint: disable=unused-argument
      self.weights = np.random.rand(*shape)
      return self.weights

    def _get_quantized_weights(shape, dtype):  # pylint: disable=unused-argument
      assert tuple(shape) == self.weights.shape

      return tf.fake_quant_with_min_max_vars(
          self.weights, -6.0, 6.0, num_bits=self.quant_params['num_bits'],
          narrow_range=self.quant_params['narrow_range'])

    quantized_model = keras.Sequential([
        QuantizeEmulateWrapper(
            layer(kernel_initializer=_get_random_weights, **kwargs),
            input_shape=input_shape,
            **self.quant_params)
    ])
    # `model` gets constructed with same parameters as `quantized_model`. The
    # weights used are a quantized version of weights used in `quantized_model`.
    # This ensures the results of both the models should be the same since
    # quantization has been applied externally to `model`.
    model = keras.Sequential([
        layer(
            input_shape=input_shape,
            kernel_initializer=_get_quantized_weights,
            **kwargs)
    ])

    inputs = np.random.rand(1, *input_shape)
    self.assertAllClose(model.predict(inputs), quantized_model.predict(inputs))


if __name__ == '__main__':
  test.main()
