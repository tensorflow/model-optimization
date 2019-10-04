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
"""ConvBatchNorm layer tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

from tensorflow.python import keras
from tensorflow.python.keras import activations
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import utils
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm

_ConvBatchNorm2D = conv_batchnorm._ConvBatchNorm2D


class ConvBatchNorm2DTest(test.TestCase):

  def setUp(self):
    super(ConvBatchNorm2DTest, self).setUp()
    self.batch_size = 8
    self.model_params = {
        'filters': 2,
        'kernel_size': (3, 3),
        'input_shape': (10, 10, 3),
        'batch_size': self.batch_size,
    }

  def _get_folded_batchnorm_model(self,
                                  is_quantized=False,
                                  post_bn_activation=None):
    return tf.keras.Sequential([
        _ConvBatchNorm2D(
            kernel_initializer=keras.initializers.glorot_uniform(seed=0),
            is_quantized=is_quantized,
            post_activation=post_bn_activation,
            **self.model_params)
    ])

  @staticmethod
  def _compute_quantization_params(model):
    # TODO(alanchiao): remove this once the converter for training-time
    # quantization supports producing a TFLite model with a float output.
    #
    # Derived from Nudge function in
    # tensorflow/core/kernels/fake_quant_ops_functor.h.
    min_val = keras.backend.eval(
        model.layers[0].post_activation._min_post_activation)
    max_val = keras.backend.eval(
        model.layers[0].post_activation._max_post_activation)
    quant_min_float = 0
    quant_max_float = 255

    scale = (max_val - min_val) / (quant_max_float - quant_min_float)
    zero_point = round(quant_min_float - min_val / scale)

    return scale, zero_point

  def _test_equivalent_to_tflite(self, model, is_tflite_quantized=False):
    _, keras_file = tempfile.mkstemp('.h5')
    _, tflite_file = tempfile.mkstemp('.tflite')

    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    model.fit(
        np.random.uniform(0, 1, size=[self.batch_size, 10, 10, 3]),
        np.random.uniform(0, 10, size=[self.batch_size, 8, 8, 2]),
        epochs=1,
        callbacks=[])

    # Prepare for inference.
    inp = np.random.uniform(0, 1, size=[self.batch_size, 10, 10, 3])
    inp = inp.astype(np.float32)

    # TensorFlow inference.
    tf_out = model.predict(inp)

    if is_tflite_quantized:
      scale, zero_point = self._compute_quantization_params(model)

      # TFLite input needs to be quantized.
      inp = inp * 255
      inp = inp.astype(np.uint8)

    # TensorFlow Lite inference.
    tf.keras.models.save_model(model, keras_file)
    with quantize.quantize_scope():
      utils.convert_keras_to_tflite(
          keras_file,
          tflite_file,
          custom_objects={'_ConvBatchNorm2D': _ConvBatchNorm2D},
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

  def testEquivalentToNonFoldedBatchNorm(self):
    folded_model = self._get_folded_batchnorm_model(is_quantized=False)

    non_folded_model = tf.keras.Sequential([
        keras.layers.Conv2D(
            kernel_initializer=keras.initializers.glorot_uniform(seed=0),
            use_bias=False,
            **self.model_params),
        keras.layers.BatchNormalization(axis=-1),
    ])

    for _ in range(2):
      inp = np.random.uniform(0, 10, size=[1, 10, 10, 3])
      folded_out = folded_model.predict(inp)
      non_folded_out = non_folded_model.predict(inp)

      # Taken from testFoldFusedBatchNorms from
      # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_test.py#L230
      self.assertAllClose(folded_out, non_folded_out, rtol=1e-04, atol=1e-06)

  def testEquivalentToFloatTFLite(self):
    model = self._get_folded_batchnorm_model(is_quantized=False)
    self._test_equivalent_to_tflite(model)

  def testQuantizedEquivalentToFloatTFLite(self):
    model = self._get_folded_batchnorm_model(is_quantized=True)
    self._test_equivalent_to_tflite(model)

  def testQuantizedWithReLUEquivalentToFloatTFLite(self):
    model = self._get_folded_batchnorm_model(
        is_quantized=True, post_bn_activation=activations.get('relu'))
    self._test_equivalent_to_tflite(model)

  def testQuantizedWithSoftmaxEquivalentToFloatTfLite(self):
    model = self._get_folded_batchnorm_model(
        is_quantized=True, post_bn_activation=activations.get('softmax'))
    self._test_equivalent_to_tflite(model)

  def testQuantizedEquivalentToQuantizedTFLite(self):
    model = self._get_folded_batchnorm_model(is_quantized=True)
    self._test_equivalent_to_tflite(model, is_tflite_quantized=True)


if __name__ == '__main__':
  test.main()
