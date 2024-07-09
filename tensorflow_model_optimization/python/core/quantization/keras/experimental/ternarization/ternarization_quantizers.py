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
"""Quantizers specific to default ternarization behavior."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantizers


def _AbsR(weights, alpha):
  """Our novel regularizer function."""
  abs_weights = tf.abs(weights)

  sparse_factor = alpha * abs_weights
  scaled_weights = 2.0 * abs_weights - 1.0

  ternary_factor = 1.0 - tf.square(tf.square(scaled_weights))
  return tf.reduce_sum(sparse_factor + ternary_factor)


def WdrLoss(weight_matrix, lambda1, alpha):
  return lambda1 * _AbsR(weight_matrix, alpha)


class TernarizationWeightsQuantizer(tf.keras.layers.Layer,
                                    quantizers.Quantizer):
  """Default ternarization quantizer."""

  def get_config(self):
    return {
        'lambdas': [1e-9, 1e-5, 1e-2],
        'steps': [7000, 10000],
        'alpha': 0.0,
    }

  def build(self, tensor_shape, name=None, layer=None):
    super(TernarizationWeightsQuantizer, self).build(tensor_shape)

    configs = self.get_config()
    self.lambda_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        values=configs['lambdas'], boundaries=configs['steps'])

    step = layer.add_weight(
        name='regularizer/step',
        initializer='zeros',
        dtype=tf.int32,
        trainable=False)

    beta = layer.add_weight(
        name='tanh/beta',
        shape=tensor_shape[-1:].as_list(),
        initializer=tf.keras.initializers.Constant(0.02),
        dtype=layer.dtype,
        trainable=True)

    return {'step': step, 'beta': beta}

  def call(self, inputs, training, weights, layer, **kwargs):
    if not training:
      return inputs

    with tf.name_scope('compute_weights'):
      tanh_kernel = tf.tanh(inputs) * weights['beta']
      weights['step'].assign_add(1)
      configs = self.get_config()
      layer.add_loss(
          WdrLoss(
              tanh_kernel,
              lambda1=self.lambda_fn(weights['step']),
              alpha=configs['alpha']))
      return tanh_kernel


class TernarizationConvTransposeWeightsQuantizer(TernarizationWeightsQuantizer):
  """Quantizer for handling weights in Conv2DTranspose layers."""

  def build(self, tensor_shape, name=None, layer=None):
    super(TernarizationConvTransposeWeightsQuantizer,
          self).build(tensor_shape, name, layer)

    configs = self.get_config()
    self.lambda_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        values=configs['lambdas'], boundaries=configs['steps'])

    step = layer.add_weight(
        name='regularizer/step',
        initializer='zeros',
        dtype=tf.int32,
        trainable=False)

    beta = layer.add_weight(
        name='tanh/beta',
        shape=tensor_shape[-2:-1].as_list(),
        initializer=tf.keras.initializers.Constant(0.02),
        dtype=layer.dtype,
        trainable=True)

    return {'step': step, 'beta': beta}

  def __call__(self, inputs, training, weights, **kwargs):
    outputs = tf.transpose(inputs, (0, 1, 3, 2))
    outputs = super(TernarizationConvTransposeWeightsQuantizer,
                    self).__call__(outputs, training, weights, **kwargs)
    outputs = tf.transpose(outputs, (0, 1, 3, 2))
    return outputs
