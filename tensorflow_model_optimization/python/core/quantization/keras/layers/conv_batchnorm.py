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
"""Convolution with folded batch normalization layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


class _ConvBatchNorm2D(Conv2D):
  """Layer for emulating the folding of batch normalization into Conv during serving.

  Implements the emulation, as described in https://arxiv.org/abs/1712.05877.
  Note that in the
  emulated form, there are two convolutions for each convolution in the original
  model.
  """

  # TODO(alanchiao): remove these defaults since in practice,
  # they will be provided by the unfolded layers. Removing the defaults
  # makes the "use_bias=False" stand out more.
  #
  # Note: use_bias is not a parameter even though it is a parameter for Conv2D.
  # This is because a Conv2D bias would be redundant with
  # BatchNormalization's bias.
  def __init__(
      self,
      # Conv2D params
      filters,
      kernel_size,
      strides=(1, 1),
      padding='valid',
      data_format=None,
      dilation_rate=(1, 1),
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      # BatchNormalization params
      axis=-1,
      momentum=0.99,
      epsilon=1e-3,
      center=True,
      scale=True,
      beta_initializer='zeros',
      gamma_initializer='ones',
      moving_mean_initializer='zeros',
      moving_variance_initializer='ones',
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_constraint=None,
      gamma_constraint=None,
      renorm=False,
      renorm_clipping=None,
      renorm_momentum=0.99,
      fused=None,
      trainable=True,
      virtual_batch_size=None,
      adjustment=None,
      name=None,
      **kwargs):
    super(_ConvBatchNorm2D, self).__init__(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)

    self.batchnorm = BatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=fused,
        trainable=trainable,
        virtual_batch_size=virtual_batch_size,
        adjustment=adjustment,
        name=name,
    )

  def build(self, input_shape):
    # responsible for trainable self.kernel weights
    super(_ConvBatchNorm2D, self).build(input_shape)

    # resposible for trainable gamma and beta weights
    self.batchnorm.build(self.compute_output_shape(input_shape))

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    conv_out = super(_ConvBatchNorm2D, self).call(inputs)

    # Not all the computations in the batchnorm need to happen,
    # but this avoids duplicating code (e.g. moving_average).
    self.batchnorm.call(conv_out)

    folded_conv_kernel_multiplier = self.batchnorm.gamma * math_ops.rsqrt(
        self.batchnorm.moving_variance + self.batchnorm.epsilon)
    folded_conv_kernel = math_ops.mul(
        folded_conv_kernel_multiplier, self.kernel, name='folded_conv_kernel')

    folded_conv_bias = math_ops.subtract(
        self.batchnorm.beta,
        self.batchnorm.moving_mean * folded_conv_kernel_multiplier,
        name='folded_conv_bias')

    # Second convolution doesn't need new trainable weights, so we
    # cannot reuse Conv2D layer.
    # TODO(alanchiao):
    # 1. See if we can at least reuse the bias logic.
    # 2. See if we need to fork between conv2d and conv2d_v2 for
    #    TensorFlow 1.XX and 2.XX.

    # Taken from keras/layers/convolutional.py:183
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()

    folded_conv_out = nn_ops.conv2d(
        inputs,
        folded_conv_kernel,
        strides=self.strides,
        padding=op_padding,
        data_format=conv_utils.convert_data_format(self.data_format,
                                                   self.rank + 2),
        dilations=self.dilation_rate,
        name='folded_conv_out',
    )

    # Taken from keras/layers/convolutional.py:200
    if self.data_format == 'channels_first':
      if self.rank == 1:
        # nn.bias_add does not accept a 1D input tensor.
        bias = array_ops.reshape(folded_conv_bias, (1, self.filters, 1))
        folded_conv_out += bias
      else:
        outputs = nn.bias_add(
            folded_conv_out, folded_conv_bias, data_format='NCHW')
    else:
      outputs = nn.bias_add(
          folded_conv_out, folded_conv_bias, data_format='NHWC')

    return outputs

  def get_config(self):
    conv_config = super(_ConvBatchNorm2D, self).get_config()
    batchnorm_config = self.batchnorm.get_config()
    return dict(list(conv_config.items()) + list(batchnorm_config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()
    # use_bias is not an argument of this class, as explained by
    # comment in __init__.
    config.pop('use_bias')

    return cls(**config)
