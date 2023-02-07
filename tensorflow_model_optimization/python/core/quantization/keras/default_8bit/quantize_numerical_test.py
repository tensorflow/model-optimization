# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Numerical verification tests for QAT."""


import tempfile

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import utils


class QuantizeNumericalTest(tf.test.TestCase, parameterized.TestCase):

  def _batch(self, dims, batch_size):
    if dims[0] is None:
      dims[0] = batch_size
    return dims

  def _create_test_data(self, model):
    x = np.random.randn(
        *self._batch(model.input.get_shape().as_list(), 1)).astype('float32')
    y = np.random.randn(
        *self._batch(model.output.get_shape().as_list(), 1)).astype('float32')

    return x, y

  def _execute_tflite(self, tflite_file, x_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    for x, _ in zip(x_test, y_test):
      x = x.reshape((1,) + x.shape)
      interpreter.set_tensor(input_index, x)
      interpreter.invoke()
      y_ = interpreter.get_tensor(output_index)

    return y_

  def _get_single_conv_model(self):
    i = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(2, kernel_size=(3, 3), strides=(2, 2))(i)
    return tf.keras.Model(i, x)

  def _get_single_dense_model(self):
    i = tf.keras.Input(shape=(5,))
    x = tf.keras.layers.Dense(3)(i)
    return tf.keras.Model(i, x)

  def _get_single_conv_relu_model(self):
    i = tf.keras.Input(shape=(6, 6, 3))
    x = tf.keras.layers.Conv2D(
        2, kernel_size=(3, 3), strides=(2, 2), activation='relu')(i)
    x = tf.keras.layers.ReLU()(x)
    return tf.keras.Model(i, x)

  def _get_stacked_convs_model(self):
    i = tf.keras.Input(shape=(64, 64, 3))
    x = tf.keras.layers.Conv2D(
        10, kernel_size=(3, 3), strides=(1, 1), activation='relu')(i)
    x = tf.keras.layers.Conv2D(
        # Setting strides to (1, 1) passes test, (2, 2) fails test?
        # Somehow one value is at border.
        # Train over 100 epochs, and issue goes away.
        # Why are all the first values zero?
        10, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        10, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        5, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        2, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    return tf.keras.Model(i, x)

  def _get_conv_bn_relu_model(self):
    i = tf.keras.Input(shape=(6, 6, 3))
    x = tf.keras.layers.Conv2D(3, kernel_size=(3, 3), strides=(2, 2))(i)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return tf.keras.Model(i, x)

  def _get_depthconv_bn_relu_model(self):
    i = tf.keras.Input(shape=(6, 6, 3))
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2))(i)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return tf.keras.Model(i, x)

  def _get_separable_conv2d_model(self):
    i = tf.keras.Input(shape=(12, 12, 3))
    x = tf.keras.layers.SeparableConv2D(
        filters=5, kernel_size=(3, 3), strides=(2, 2))(i)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return tf.keras.Model(i, x)

  def _get_sepconv1d_bn_relu_model(self):
    i = tf.keras.Input(shape=(8, 3))
    x = tf.keras.layers.SeparableConv1D(
        filters=5, kernel_size=3, strides=2)(i)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return tf.keras.Model(i, x)

  def _get_sepconv1d_bn_model(self):
    i = tf.keras.Input(shape=(8, 3))
    x = tf.keras.layers.SeparableConv1D(
        filters=5, kernel_size=3, strides=2)(i)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.Model(i, x)

  def _get_sepconv1d_stacked_model(self):
    i = tf.keras.Input(shape=(8, 3))
    x = tf.keras.layers.SeparableConv1D(
        filters=5, kernel_size=3, strides=2)(i)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SeparableConv1D(
        filters=5, kernel_size=3, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return tf.keras.Model(i, x)

  def _get_upsampling2d_nearest_model(self):
    i = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.UpSampling2D(size=(3, 4), interpolation='nearest')(i)
    return tf.keras.Model(i, x)

  def _get_upsampling2d_bilinear_model(self):
    i = tf.keras.Input(shape=(1, 3, 1))
    x = tf.keras.layers.UpSampling2D(size=(1, 5), interpolation='bilinear')(i)
    return tf.keras.Model(i, x)

  def _get_conv2d_transpose_model(self):
    i = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2DTranspose(
        2, kernel_size=(3, 3), strides=(2, 2))(
            i)
    return tf.keras.Model(i, x)

  @parameterized.parameters([
      _get_single_conv_model, _get_single_dense_model,
      _get_single_conv_relu_model, _get_stacked_convs_model,
      _get_conv_bn_relu_model, _get_depthconv_bn_relu_model,
      _get_separable_conv2d_model,
      _get_sepconv1d_bn_model, _get_sepconv1d_bn_relu_model,
      _get_sepconv1d_stacked_model,
      _get_upsampling2d_nearest_model,
      # _get_upsampling2d_bilinear_model
      # TODO(tfmot): There are gaps between ResizeBilinear with FakeQuant and
      # TFLite quantized ResizeBilinear op. It has a bit more quantization
      # error than other ops in this test now.
      _get_conv2d_transpose_model,
  ])
  def testModelEndToEnd(self, model_fn):
    # 1. Check whether quantized model graph can be constructed.
    model = model_fn(self)
    model = quantize.quantize_model(model)

    # 2. Sanity check to ensure basic training on random data works.
    x_train, y_train = self._create_test_data(model)
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100)

    x_test, y_test = self._create_test_data(model)

    y_tf = model.predict(x_test)

    # 3. Ensure conversion to TFLite works.
    _, tflite_file = tempfile.mkstemp('.tflite')
    print('TFLite File: ', tflite_file)
    with quantize.quantize_scope():
      utils.convert_keras_to_tflite(model, tflite_file)

    # 4. Verify input runs on converted model.
    y_tfl = self._execute_tflite(tflite_file, x_test, y_test)

    # 5. Verify results are the same in TF and TFL.
    # TODO(pulkitb): Temporarily raise tolerances since some rounding
    # changes in x86 kernels are causing values to differ by 'scale'.
    self.assertAllClose(y_tf, y_tfl, atol=1e-1, rtol=1e-1)


if __name__ == '__main__':
  if hasattr(tf.keras.__internal__, 'enable_unsafe_deserialization'):
    tf.keras.__internal__.enable_unsafe_deserialization()
  tf.test.main()
