# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Registry responsible for built-in keras classes."""

from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import DepthwiseConv2D
from tensorflow.python.keras.layers import SeparableConv2D


class QuantizeEmulateRegistry(object):
  """Registry responsible for built-in keras classes."""

  # The keys represent built-in keras layers and the values represent the
  # the variables within the layers which hold the kernel weights. This
  # allows the wrapper to access and modify the weights.
  _CONVERTER_MAP = {
      Dense: ['kernel'],
      Conv2D: ['kernel'],
      SeparableConv2D: ['depthwise_kernel', 'pointwise_kernel'],
      DepthwiseConv2D: ['depthwise_kernel']
  }

  # TODO(pulkitb): Turn this into a singleton instead of using classmethods.

  @classmethod
  def supports(cls, layer):
    """Returns whether the registry supports this layer type.

    Args:
      layer: The layer to check for support.

    Returns:
      True/False whether the layer type is supported.

    """
    return layer.__class__ in cls._CONVERTER_MAP

  @classmethod
  def _weight_names(cls, layer):
    return cls._CONVERTER_MAP[layer.__class__]

  @classmethod
  def make_quantizable(cls, layer):
    """Modifies a built-in layer object to support quantize emulation.

    Args:
      layer: layer to modify for support.

    Returns:
      The modified layer object.

    """

    if not cls.supports(layer):
      raise ValueError('Layer ' + str(layer.__class__) + ' is not supported.')

    def get_quantizable_weights():
      quantizable_weights = []
      for weight_name in cls._weight_names(layer):
        quantizable_weights.append(getattr(layer, weight_name))
      return quantizable_weights

    def set_quantizable_weights(weights):
      """Function gets augmented to layers to support quantize emulation.

      Arguments:
          weights: a list of Numpy arrays. The number
              of arrays and their shape must match
              number of the dimensions of the weights
              of the layer (i.e. it should match the
              output of `get_quantizable_weights`).

      Raises:
          ValueError: If the provided weights list does not match the
              layer's specifications.

      """
      existing_weights = get_quantizable_weights()

      if len(existing_weights) != len(weights):
        raise ValueError('`set_quantizable_weights` called on layer {} with {} '
                         'parameters, but layer expects {}.'.format(
                             layer.name, len(existing_weights), len(weights)))

      for ew, w in zip(existing_weights, weights):
        if ew.shape != w.shape:
          raise ValueError('Layer weight shape {} incompatible with provided '
                           'weight shape {}'.format(ew.shape, w.shape))

      for weight_name, weight in zip(cls._weight_names(layer), weights):
        setattr(layer, weight_name, weight)

    layer.get_quantizable_weights = get_quantizable_weights
    layer.set_quantizable_weights = set_quantizable_weights

    return layer
