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
"""Quantize Annotate Wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.layers.wrappers import Wrapper


class QuantizeAnnotate(Wrapper):
  """Annotates layers which quantization should be applied to.

  QuantizeAnnotate does not actually apply quantization to the underlying
  layers but acts as a way to specify which layers quantization should be
  applied to.

  The wrapper functions as a NoOp or pass-through wrapper by simply delegating
  calls to the underlying layer. The presence of this wrapper indicates to code
  which actually applies quantization to determine which layers should be
  modified.
  """

  def __init__(self,
               layer,
               num_bits,
               narrow_range=True,
               symmetric=True,
               **kwargs):
    """Create a quantize annotate wrapper over a keras layer.

    Args:
      layer: The keras layer to be quantized.
      num_bits: Number of bits for quantization
      narrow_range: Whether to use the narrow quantization range [1; 2^num_bits
        - 1] or wide range [0; 2^num_bits - 1].
      symmetric: If true, use symmetric quantization limits instead of training
        the minimum and maximum of each quantization range separately.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    super(QuantizeAnnotate, self).__init__(layer, **kwargs)

    self._num_bits = num_bits
    self._narrow_range = narrow_range
    self._symmetric = symmetric

  def call(self, inputs, training=None):
    return self.layer.call(inputs)

  def get_quantize_params(self):
    return {
        'num_bits': self._num_bits,
        'symmetric': self._symmetric,
        'narrow_range': self._narrow_range
    }

  def get_config(self):
    base_config = super(QuantizeAnnotate, self).get_config()
    config = self.get_quantize_params()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()

    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    layer = deserialize_layer(config.pop('layer'))
    config['layer'] = layer

    return cls(**config)

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(input_shape)

  @property
  def trainable(self):
    return self.layer.trainable

  @trainable.setter
  def trainable(self, value):
    self.layer.trainable = value

  @property
  def trainable_weights(self):
    return self.layer.trainable_weights

  @property
  def non_trainable_weights(self):
    return self.layer.non_trainable_weights + self._non_trainable_weights

  @property
  def updates(self):
    return self.layer.updates + self._updates

  @property
  def losses(self):
    return self.layer.losses + self._losses

  def get_weights(self):
    return self.layer.get_weights()

  def set_weights(self, weights):
    self.layer.set_weights(weights)
