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
"""ConvBatchNorm layer tests.

See FoldedBatchNormTest for shared test cases between different
classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import numpy as np
from six.moves import range
import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras import activations
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import utils
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm_test_utils

_ConvBatchNorm2D = conv_batchnorm._ConvBatchNorm2D
_DepthwiseConvBatchNorm2D = conv_batchnorm._DepthwiseConvBatchNorm2D
Conv2DModel = conv_batchnorm_test_utils.Conv2DModel
DepthwiseConv2DModel = conv_batchnorm_test_utils.DepthwiseConv2DModel


class FoldedBatchNormTestBase(test.TestCase):

  @staticmethod
  def _compute_quantization_params(model):
    # TODO(alanchiao): remove this once the converter for training-time
    # quantization supports producing a TFLite model with a float output.
    #
    # Derived from Nudge function in
    # tensorflow/core/kernels/fake_quant_ops_functor.h.
    min_val = keras.backend.eval(model.layers[0]._activation_min_var)
    max_val = keras.backend.eval(model.layers[0]._activation_max_var)
    quant_min_float = 0
    quant_max_float = 255

    scale = (max_val - min_val) / (quant_max_float - quant_min_float)
    zero_point = round(quant_min_float - min_val / scale)

    return scale, zero_point

  # This does a basic serialize/deserialize test since deserialization
  # occurs during conversion to TFLite.
  def _test_equal_tf_and_tflite_outputs(self,
                                        tf_model,
                                        is_tflite_quantized=False):
    _, keras_file = tempfile.mkstemp('.h5')
    _, tflite_file = tempfile.mkstemp('.tflite')

    batched_input_shape = self._get_batched_input_shape()
    output_shape = self._get_output_shape()

    tf_model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    tf_model.fit(
        np.random.uniform(0, 1, size=batched_input_shape),
        np.random.uniform(0, 10, size=output_shape),
        epochs=1,
        callbacks=[])
    # Prepare for inference.
    inp = np.random.uniform(0, 1, size=batched_input_shape)
    inp = inp.astype(np.float32)

    # TensorFlow inference.
    tf_out = tf_model.predict(inp)

    if is_tflite_quantized:
      scale, zero_point = self._compute_quantization_params(tf_model)

      # TFLite input needs to be quantized.
      inp = inp * 255
      inp = inp.astype(np.uint8)

    # TensorFlow Lite inference.
    tf.keras.models.save_model(tf_model, keras_file)
    with quantize.quantize_scope():
      utils.convert_keras_to_tflite(
          keras_file,
          tflite_file,
          custom_objects={
              '_ConvBatchNorm2D': _ConvBatchNorm2D,
              '_DepthwiseConvBatchNorm2D': _DepthwiseConvBatchNorm2D
          },
          is_quantized=is_tflite_quantized)

    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()
    tflite_out = interpreter.get_tensor(output_index)

    if is_tflite_quantized:
      # dequantize outputs
      tflite_out = [scale * (x - zero_point) for x in tflite_out]
      # Off by 1 in quantized output. Notably we cannot reduce this. There is
      # an existing mismatch between TensorFlow and TFLite (from
      # contrib.quantize days).
      self.assertAllClose(tf_out, tflite_out, atol=scale)
    else:
      # Taken from testFoldFusedBatchNorms from
      # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_test.py#L230
      self.assertAllClose(tf_out, tflite_out, rtol=1e-04, atol=1e-06)

  def _test_equal_outputs(self, model, model2):
    for _ in range(2):
      inp = np.random.uniform(0, 10, size=self._get_batched_input_shape())
      model_out = model.predict(inp)
      model2_out = model2.predict(inp)

      # Taken from testFoldFusedBatchNorms from
      # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_test.py#L230
      self.assertAllClose(model_out, model2_out, rtol=1e-04, atol=1e-06)


class ConvBatchNorm2DTest(FoldedBatchNormTestBase):

  def _get_folded_batchnorm_model(self,
                                  is_quantized=False,
                                  post_bn_activation=None):
    return Conv2DModel.get_folded_batchnorm_model(
        is_quantized=is_quantized, post_bn_activation=post_bn_activation)

  def _get_nonfolded_batchnorm_model(self):
    return Conv2DModel.get_nonfolded_batchnorm_model()

  def _get_batched_input_shape(self):
    return Conv2DModel.get_batched_input_shape()

  def _get_output_shape(self):
    return Conv2DModel.get_output_shape()

  def testEquivalentToNonFoldedBatchNorm(self):
    self._test_equal_outputs(
        self._get_folded_batchnorm_model(is_quantized=False),
        self._get_nonfolded_batchnorm_model())

  def testEquivalentToFloatTFLite(self):
    tf_model = self._get_folded_batchnorm_model(is_quantized=False)
    self._test_equal_tf_and_tflite_outputs(tf_model)

  def testQuantizedEquivalentToFloatTFLite(self):
    tf_model = self._get_folded_batchnorm_model(is_quantized=True)
    self._test_equal_tf_and_tflite_outputs(tf_model)

  def testQuantizedWithReLUEquivalentToFloatTFLite(self):
    tf_model = self._get_folded_batchnorm_model(
        is_quantized=True, post_bn_activation=activations.get('relu'))
    self._test_equal_tf_and_tflite_outputs(tf_model)

  def testQuantizedWithAdvancedReLUEquivalentToFloatTFLite(self):
    tf_model = self._get_folded_batchnorm_model(
        is_quantized=True, post_bn_activation=keras.layers.ReLU(max_value=6.0))
    self._test_equal_tf_and_tflite_outputs(tf_model)

  def testQuantizedWithSoftmaxEquivalentToFloatTfLite(self):
    tf_model = self._get_folded_batchnorm_model(
        is_quantized=True, post_bn_activation=activations.get('softmax'))
    self._test_equal_tf_and_tflite_outputs(tf_model)

  def testQuantizedEquivalentToQuantizedTFLite(self):
    tf_model = self._get_folded_batchnorm_model(is_quantized=True)
    self._test_equal_tf_and_tflite_outputs(tf_model, is_tflite_quantized=True)


class DepthwiseConvBatchNorm2DTest(FoldedBatchNormTestBase):

  def _get_folded_batchnorm_model(self,
                                  is_quantized=False,
                                  post_bn_activation=None):
    return DepthwiseConv2DModel.get_folded_batchnorm_model(
        is_quantized=is_quantized, post_bn_activation=post_bn_activation)

  def _get_nonfolded_batchnorm_model(self):
    return DepthwiseConv2DModel.get_nonfolded_batchnorm_model()

  def _get_batched_input_shape(self):
    return DepthwiseConv2DModel.get_batched_input_shape()

  def _get_output_shape(self):
    return DepthwiseConv2DModel.get_output_shape()

  def testEquivalentToNonFoldedBatchNorm(self):
    self._test_equal_outputs(
        self._get_folded_batchnorm_model(is_quantized=False),
        self._get_nonfolded_batchnorm_model())

  def testEquivalentToFloatTFLite(self):
    tf_model = self._get_folded_batchnorm_model(is_quantized=False)
    self._test_equal_tf_and_tflite_outputs(tf_model)

  def testQuantizedEquivalentToFloatTFLite(self):
    tf_model = self._get_folded_batchnorm_model(is_quantized=True)
    self._test_equal_tf_and_tflite_outputs(tf_model)

  def testQuantizedWithSoftmaxEquivalentToFloatTfLite(self):
    tf_model = self._get_folded_batchnorm_model(
        is_quantized=True, post_bn_activation=activations.get('softmax'))
    self._test_equal_tf_and_tflite_outputs(tf_model)

  def testQuantizedWithReLUEquivalentToFloatTFLite(self):
    tf_model = self._get_folded_batchnorm_model(
        is_quantized=True, post_bn_activation=activations.get('relu'))
    self._test_equal_tf_and_tflite_outputs(tf_model)

  def testQuantizedWithAdvancedReLUEquivalentToFloatTFLite(self):
    tf_model = self._get_folded_batchnorm_model(
        is_quantized=True, post_bn_activation=keras.layers.ReLU(max_value=6.0))
    self._test_equal_tf_and_tflite_outputs(tf_model)

  def testQuantizedEquivalentToQuantizedTFLite(self):
    tf_model = self._get_folded_batchnorm_model(is_quantized=True)
    self._test_equal_tf_and_tflite_outputs(tf_model, is_tflite_quantized=True)


if __name__ == '__main__':
  test.main()
