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

from tensorflow.python.keras.layers import deserialize as deserialize_layer
from tensorflow.python.keras.layers.wrappers import Wrapper
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object


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

  _UNSUPPORTED_LAYER_ERROR_MSG = (
      'Layer {} not supported for quantization. Layer should either inherit '
      'QuantizeEmulatableLayer or be a supported keras built-in layer.')

  def __init__(self,
               layer,
               quantize_provider=None,
               **kwargs):
    """Create a quantize annotate wrapper over a keras layer.

    Args:
      layer: The keras layer to be quantized.
      quantize_provider: `QuantizeProvider` to quantize layer.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    super(QuantizeAnnotate, self).__init__(layer, **kwargs)

    self.quantize_provider = quantize_provider

  def call(self, inputs, training=None):
    return self.layer.call(inputs)

  def get_config(self):
    base_config = super(QuantizeAnnotate, self).get_config()
    config = {
        'quantize_provider': serialize_keras_object(self.quantize_provider)
    }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()

    # QuantizeAnnotate may be constructed with any QuantizeProvider and the
    # wrapper itself cannot know all the possible provider classes.
    # The deserialization code should ensure the QuantizeProvider is in keras
    # serialization scope.
    # TODO(pulkitb): Decide if built-in QuantizeProviders should be in scope.
    quantize_provider = deserialize_keras_object(
        config.pop('quantize_provider'),
        module_objects=globals(),
        custom_objects=None)

    layer = deserialize_layer(config.pop('layer'))

    return cls(layer=layer, quantize_provider=quantize_provider, **config)

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
