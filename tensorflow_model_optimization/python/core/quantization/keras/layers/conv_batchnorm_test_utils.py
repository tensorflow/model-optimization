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
"""Test utils for conv batchnorm folding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

keras = tf.keras


def _get_conv2d_params():
  return {
      'kernel_size': (3, 3),
      'input_shape': (10, 10, 3),
      'batch_size': 8,
  }


def _get_initializer(random_init):
  if random_init:
    kernel_initializer = keras.initializers.glorot_uniform()
  else:
    kernel_initializer = keras.initializers.glorot_uniform(seed=0)
  return kernel_initializer


class Conv2DModel(object):
  """Construct and access Conv + (Squeeze) + BatchNorm + activation models."""

  params = {
      'filters': 2,
      'kernel_size': (1, 1),
      'input_shape': (1, 3, 3),
      'batch_size': 1,
  }

  @classmethod
  def get_batched_input_shape(cls):
    """Return input shape with batch size."""
    shape = [cls.params['batch_size']]
    shape.extend(cls.params['input_shape'])
    return shape

  @classmethod
  def get_output_shape(cls):
    return [cls.params['batch_size'], 2, 2, 2]

  @classmethod
  def get_nonfolded_batchnorm_model(cls,
                                    post_bn_activation=None,
                                    model_type='sequential',
                                    random_init=False,
                                    squeeze_type=False,
                                    normalization_type='BatchNormalization'):
    """Return nonfolded Conv2D + BN + optional activation model."""
    if normalization_type == 'BatchNormalization':
      normalization = keras.layers.BatchNormalization
    elif normalization_type == 'SyncBatchNormalization':
      normalization = keras.layers.experimental.SyncBatchNormalization

    if squeeze_type == 'sepconv1d_squeeze':
      squeeze_layer = tf.keras.layers.Lambda(
          lambda x: tf.squeeze(x, [1]), name='sepconv1d_squeeze_1')
    else:
      squeeze_layer = None

    if model_type == 'sequential':
      layers = []
      layers.append(
          keras.layers.Conv2D(
              kernel_initializer=_get_initializer(random_init),
              use_bias=False,
              **cls.params))
      if squeeze_layer is not None:
        layers.append(squeeze_layer)
      layers.append(normalization(axis=-1))
      if post_bn_activation is not None:
        layers += post_bn_activation
      return tf.keras.Sequential(layers)
    else:
      inp = keras.layers.Input(cls.params['input_shape'],
                               cls.params['batch_size'])
      x = keras.layers.Conv2D(
          cls.params['filters'],
          cls.params['kernel_size'],
          kernel_initializer=_get_initializer(random_init),
          use_bias=False)(
              inp)
      if squeeze_layer is not None:
        x = squeeze_layer(x)
      out = normalization(axis=-1)(x)
      if post_bn_activation is not None:
        out = post_bn_activation(out)
      return tf.keras.Model(inp, out)


class DepthwiseConv2DModel(Conv2DModel):
  """Construct and access DWConv + (Squeeze) + BatchNorm + activation models."""

  params = {
      'kernel_size': (1, 1),
      'input_shape': (1, 10, 3),
      'batch_size': 8,
  }

  @classmethod
  def get_output_shape(cls):
    return [cls.params['batch_size'], 8, 8, 3]

  @classmethod
  def get_nonfolded_batchnorm_model(cls,
                                    post_bn_activation=None,
                                    model_type='sequential',
                                    random_init=False,
                                    squeeze_type=False,
                                    normalization_type='BatchNormalization'):
    if normalization_type == 'BatchNormalization':
      normalization = keras.layers.BatchNormalization
    elif normalization_type == 'SyncBatchNormalization':
      normalization = keras.layers.experimental.SyncBatchNormalization

    if squeeze_type == 'sepconv1d_squeeze':
      squeeze_layer = tf.keras.layers.Lambda(
          lambda x: tf.squeeze(x, [1]), name='sepconv1d_squeeze_1')
    else:
      squeeze_layer = None

    if model_type == 'sequential':
      layers = []
      layers.append(
          keras.layers.DepthwiseConv2D(
              depthwise_initializer=_get_initializer(random_init),
              use_bias=False,
              **cls.params))
      if squeeze_layer is not None:
        layers.append(squeeze_layer)
      layers.append(normalization(axis=-1))
      if post_bn_activation is not None:
        layers += post_bn_activation
      return tf.keras.Sequential(layers)
    else:
      inp = keras.layers.Input(cls.params['input_shape'],
                               cls.params['batch_size'])
      x = keras.layers.DepthwiseConv2D(
          cls.params['kernel_size'],
          depthwise_initializer=_get_initializer(random_init),
          use_bias=False)(
              inp)
      if squeeze_layer is not None:
        x = squeeze_layer(x)
      out = normalization(axis=-1)(x)
      if post_bn_activation is not None:
        out = post_bn_activation(out)
      return tf.keras.Model(inp, out)
