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
"""Tests for TFLite Transforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.platform import test

from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm_test_utils
from tensorflow_model_optimization.python.core.quantization.keras.tflite import tflite_transforms

ModelTransformer = model_transformer.ModelTransformer

Conv2DModel = conv_batchnorm_test_utils.Conv2DModel
DepthwiseConv2DModel = conv_batchnorm_test_utils.DepthwiseConv2DModel


class TFLiteTransformsTest(test.TestCase):

  def testTransformsConvBNReLUPattern(self):
    model = Conv2DModel.get_nonfolded_batchnorm_model(
        post_bn_activation=keras.layers.ReLU(6.0), model_type='functional')
    folded_model = Conv2DModel.get_folded_batchnorm_model(
        post_bn_activation=keras.layers.ReLU(6.0), is_quantized=True)

    with quantize.quantize_scope():
      transformed_model = ModelTransformer(
          model, [tflite_transforms.Conv2DBatchNormReLU6Fold()]).transform()

    inputs = np.random.standard_normal(Conv2DModel.get_batched_input_shape())
    self.assertAllClose(
        transformed_model.predict(inputs), folded_model.predict(inputs))

  def testTransformsDepthwiseConvBNReLUPattern(self):
    model = DepthwiseConv2DModel.get_nonfolded_batchnorm_model(
        post_bn_activation=keras.layers.ReLU(6.0), model_type='functional')
    folded_model = DepthwiseConv2DModel.get_folded_batchnorm_model(
        post_bn_activation=keras.layers.ReLU(6.0), is_quantized=True)

    with quantize.quantize_scope():
      transformed_model = ModelTransformer(
          model,
          [tflite_transforms.DepthwiseConv2DBatchNormReLU6Fold()]).transform()

    inputs = np.random.standard_normal(
        DepthwiseConv2DModel.get_batched_input_shape())
    self.assertAllClose(
        transformed_model.predict(inputs), folded_model.predict(inputs))


if __name__ == '__main__':
  test.main()
