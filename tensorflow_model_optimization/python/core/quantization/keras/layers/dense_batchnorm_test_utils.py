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
"""Test utils for dense batchnorm folding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.keras.compat import keras


class DenseModel(object):
  """Construct and access Dense + BatchNorm + activation models."""

  params = {
      'units': 32,
      'input_shape': (32,),
      'batch_size': 1,
  }

  @classmethod
  def get_batched_input_shape(cls):
    """Return input shape with batch size."""
    shape = [cls.params['batch_size']]
    shape.extend(cls.params['input_shape'])
    return shape

  @classmethod
  def get_nonfolded_batchnorm_model(cls,
                                    post_bn_activation=None,
                                    normalization_type='BatchNormalization'):
    """Return nonfolded Dense + BN + optional activation model."""
    if normalization_type == 'BatchNormalization':
      normalization = keras.layers.BatchNormalization
    elif normalization_type == 'SyncBatchNormalization':
      normalization = keras.layers.experimental.SyncBatchNormalization

    inp = keras.layers.Input(cls.params['input_shape'],
                             cls.params['batch_size'])
    x = keras.layers.Dense(cls.params['units'])(inp)
    out = normalization(axis=-1)(x)
    if post_bn_activation is not None:
      out = post_bn_activation(out)
    return keras.Model(inp, out)
