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
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.quantization.keras import utils
from tensorflow_model_optimization.python.core.quantization.keras.layers.conv_batchnorm import _ConvBatchNorm2D


class ConvBatchNorm2DTest(test.TestCase):

  def setUp(self):
    super(ConvBatchNorm2DTest, self).setUp()
    self.model_params = {
        'filters': 2,
        'kernel_size': (3, 3),
        'input_shape': (10, 10, 3),
    }

  def _get_folded_batchnorm_model(self):
    return tf.keras.Sequential([
        _ConvBatchNorm2D(
            kernel_initializer=keras.initializers.glorot_uniform(seed=0),
            **self.model_params)
    ])

  def testEquivalentToNonFoldedBatchNorm(self):
    folded_model = self._get_folded_batchnorm_model()

    non_folded_model = tf.keras.Sequential([
        keras.layers.Conv2D(
            kernel_initializer=keras.initializers.glorot_uniform(seed=0),
            use_bias=False,
            **self.model_params),
        keras.layers.BatchNormalization(axis=-1),
    ])

    for _ in range(10):
      inp = np.random.uniform(0, 10, size=[1, 10, 10, 3])
      folded_out = folded_model.predict(inp)
      non_folded_out = non_folded_model.predict(inp)

      # Taken from testFoldFusedBatchNorms from
      # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_test.py#L230
      self.assertAllClose(folded_out, non_folded_out, rtol=1e-04, atol=1e-06)

  def testEquivalentToTFLite(self):
    model = self._get_folded_batchnorm_model()

    _, keras_file = tempfile.mkstemp('.h5')
    _, tflite_file = tempfile.mkstemp('.tflite')

    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    model.fit(
        np.random.uniform(0, 1, size=[1, 10, 10, 3]),
        np.random.uniform(0, 10, size=[1, 8, 8, 2]),
        epochs=1,
        callbacks=[])

    # Prepare for inference.
    inp = np.random.uniform(0, 1, size=[1, 10, 10, 3])
    inp = inp.astype(np.float32)

    # TensorFlow inference.
    tf_out = model.predict(inp)

    # TensorFlow Lite inference.
    tf.keras.models.save_model(model, keras_file)
    utils.convert_keras_to_tflite(
        keras_file,
        tflite_file,
        custom_objects={'_ConvBatchNorm2D': _ConvBatchNorm2D},
        is_quantized=False)

    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()
    tflite_out = interpreter.get_tensor(output_index)

    # Taken from testFoldFusedBatchNorms from
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_test.py#L230
    self.assertAllClose(tf_out, tflite_out, rtol=1e-04, atol=1e-06)


if __name__ == '__main__':
  test.main()
