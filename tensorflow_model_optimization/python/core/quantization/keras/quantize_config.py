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
"""Interface for a layer to express how to quantize it."""

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class QuantizeConfig(object):
  """ABC interface for Keras layers to express how they should be quantized."""

  @abc.abstractmethod
  def get_weights_and_quantizers(self, layer):
    """Return weights to be quantized along with their quantizers.

    Args:
      layer: layer being quantized.

    Returns:
      List of 2-tuples. Each tuple is a weight tensor and an associated
      quantizer.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def get_activations_and_quantizers(self, layer):
    """Return activations to be quantized along with their quantizers.

    Args:
      layer: layer being quantized.

    Returns:
      List of 2-tuples. Each tuple is a keras activation and an associated
      quantizer.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def set_quantize_weights(self, layer, quantize_weights):
    """Replace the weights in the layer with quantized weights.

    Args:
      layer: layer being quantized.
      quantize_weights: List of quantized weight tensors.

    Returns:
      List of 2-tuples. Each tuple is a weight tensor and an associated
      quantizer.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def set_quantize_activations(self, layer, quantize_activations):
    """Replace the activations in the layer with quantized activations.

    Args:
      layer: layer being quantized.
      quantize_activations: List of `QuantizeAwareActivation`s to replace
        layer activations.

    Returns:
      List of 2-tuples. Each tuple is a keras activation and an associated
      quantizer.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def get_output_quantizers(self, layer):
    """Returns the quantizer used to quantize the outputs from a layer.

    For certain layers, we may want to quantize the outputs tensors returned
    by the layer's `call` function. This allows us to quantize those output
    tensors.

    This function should return a list of quantizers. In most cases, a layer
    outputs only a single tensor so it should only have one quantizer. Return
    an empty list for if no quantization operation is desired on the results
    of the layer.

    Args:
      layer: layer being quantized.

    Returns:
      List of `Quantizer`s to be used to quantize the resulting tensors from
      a layer.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def get_config(self):
    raise NotImplementedError('QuantizeConfig should implement get_config().')
