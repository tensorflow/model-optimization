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
"""Tests for compress wrappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_model_optimization.python.core.internal.compression.keras import compress_wrapper

keras = tf.keras
layers = keras.layers

tf.enable_v2_behavior()


class KerayLayerWrapperTest(tf.test.TestCase):

  def testDense(self):
    input_shape = (5, 5)
    layer = layers.Dense(10)
    wrapped_layer = compress_wrapper.KerasLayerWrapper(layer, ['kernel'])

    inputs = tf.ones(input_shape)
    self.assertAllEqual(layer(inputs),
                        wrapped_layer(inputs, kernel=layer.kernel))

    new_kernel = tf.random.uniform(layer.kernel.shape, -1., 1.)

    wrapped_output = wrapped_layer(inputs, kernel=new_kernel)

    new_layer = layers.Dense(10)
    new_layer.build(input_shape)
    new_layer.kernel.assign(new_kernel)
    output = new_layer(inputs)

    self.assertAllEqual(output, wrapped_output)

  def testConv2D(self):
    input_shape = (1, 10, 10, 3)
    layer = layers.Conv2D(8, (3, 3))
    wrapped_layer = compress_wrapper.KerasLayerWrapper(layer, ['kernel'])

    inputs = tf.ones(input_shape)
    self.assertAllEqual(layer(inputs),
                        wrapped_layer(inputs, kernel=layer.kernel))

    new_kernel = tf.random.uniform(layer.kernel.shape, -1., 1.)

    wrapped_output = wrapped_layer(inputs, kernel=new_kernel)

    new_layer = layers.Conv2D(8, (3, 3))
    new_layer.build(input_shape)
    new_layer.kernel.assign(new_kernel)
    output = new_layer(inputs)

    self.assertAllEqual(output, wrapped_output)


class SimpleModule(tf.Module):

  def __init__(self, name=None):
    super().__init__(name=name)
    self.multiplier = tf.Variable(2.0)

  def __call__(self, x):
    return self.multiplier * x


def simple_setter(module, name, value):
  if name == 'multiplier':
    module.multiplier = value


class ModuleWrapperTest(tf.test.TestCase):

  def testSimpleModule(self):
    wrapped_module = compress_wrapper.ModuleWrapper(
        constructor=SimpleModule,
        var_names=['multiplier'],
        setter=simple_setter)
    wrapped_module.build()

    inputs = tf.ones((2, 2))
    expected_output = inputs * 3
    output = wrapped_module.call('__call__', inputs, tf.constant([3.]))

    self.assertAllEqual(output, expected_output)


if __name__ == '__main__':
  tf.test.main()
