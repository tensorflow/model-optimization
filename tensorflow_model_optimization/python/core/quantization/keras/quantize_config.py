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

import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras import quantizers


@six.add_metaclass(abc.ABCMeta)
class QuantizeConfig(object):
  """ABC interface for Keras layers to express how they should be quantized.

  This is an experimental API not subject to backward compatibility.

  QuantizeConfig encapsulates all the information needed by the quantization
  code to quantize a layer. It specifies what parts of a layer should be
  quantized and how they should be quantized.

  It can be used to precisely control the quantization behavior of a layer.
  The framework provides default behavior for each layer, but this can be used
  to override it.

  Create QuantizeConfig for a Dense layer:

  ```python
  class MyDenseQuantizeConfig(QuantizeConfig):

    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer())]

    def get_activations_and_quantizers(self, layer):
      return [(layer.activation, MovingAverageQuantizer())]

    def set_quantize_weights(self, layer, quantize_weights):
      layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
      layer.activation = quantize_activations[0]

    def get_output_quantizers(self, layer):
      # Does not quantize output, since we return an empty list.
      return []

    def get_config(self):
      return {}
  ```

  For a full example, see
  https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide.md

  """

  @abc.abstractmethod
  def get_weights_and_quantizers(self, layer):
    """Return weights to be quantized along with their quantizers.

    This function tells the quantize code which weights within a layer
    should be quantized, and how. The weights are the TF variables in a layer
    and the quantizers are `Quantizer` instances.

    This method is invoked by the quantization code when quantizing a layer.

    Example for a `Dense` layer:
    ```python
    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer())]
    ```

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

    This function tells the quantize code which activations within a layer
    should be quantized, and how. The activations are the activation
    attributes in a layer, and the quantizers are `Quantizer` instances.

    This method is invoked by the quantization code when quantizing a layer.

    Example for a `Dense` layer:
    ```python
    def get_activations_and_quantizers(self, layer):
      return [(layer.activation, MovingAverageQuantizer())]
    ```

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

    This method is invoked by the quantization code to replace the weights
    within a layer with quantized weights. It is responsible for ensuring that
    the weights within a layer are properly replaced.

    Example for a `Dense` layer:
    ```python
    def set_quantize_weights(self, layer, quantize_weights):
      layer.kernel = quantize_weights[0]
    ```

    Args:
      layer: layer being quantized.
      quantize_weights: List of quantized weight tensors.

    Returns:
      None
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def set_quantize_activations(self, layer, quantize_activations):
    """Replace the activations in the layer with quantized activations.

    This method is invoked by the quantization code to replace the activations
    within a layer with quantized activations. It is responsible for ensuring
    that the activations within a layer are properly replaced.

    Example for a `Dense` layer:
    ```python
    def set_quantize_activations(self, layer, quantize_activations):
      layer.activation = quantize_activations[0]
    ```

    Args:
      layer: layer being quantized.
      quantize_activations: List of quantized activations to replace the
        original activations in the layer.

    Returns:
      None
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
    """Returns the config used to serialize `QuantizeConfig`."""
    raise NotImplementedError('QuantizeConfig should implement get_config().')


class OutputOnlyConfig(QuantizeConfig):
  """QuantizeConfig that only quantizes output."""

  def __init__(self, quantize_config):
    self.quantize_config = quantize_config

  def get_weights_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def get_activations_and_quantizers(self, layer):
    return self.quantize_config.get_activations_and_quantizers(layer)

  def set_quantize_activations(self, layer, quantize_activations):
    return self.quantize_config.set_quantize_activations(
        layer, quantize_activations)

  def get_output_quantizers(self, layer):
    return self.quantize_config.get_output_quantizers(layer)

  def get_config(self):
    return {'quantize_config': self.quantize_config}

  @classmethod
  def from_config(cls, config):
    return cls(**config)


class FixedQuantizeConfig(QuantizeConfig):
  """QuantizeConfig that quantizes output with fixed range."""

  def __init__(self, config, num_bits, init_min, init_max, narrow_range):
    self.config = config
    self.num_bits = num_bits
    self.init_min = init_min
    self.init_max = init_max
    self.narrow_range = narrow_range
    self.fixed_quantizer = quantizers.FixedQuantizer(
        num_bits=num_bits,
        init_min=init_min,
        init_max=init_max,
        narrow_range=narrow_range)

  def get_weights_and_quantizers(self, layer):
    return self.config.get_weights_and_quantizers(layer)

  def set_quantize_weights(self, layer, quantize_weights):
    return self.config.set_quantize_weights(layer, quantize_weights)

  def get_activations_and_quantizers(self, layer):
    activations_and_quantizers = (
        self.config.get_activations_and_quantizers(layer))
    return [(activation, self.fixed_quantizer)
            for activation, _ in activations_and_quantizers]

  def set_quantize_activations(self, layer, quantize_activations):
    return self.config.set_quantize_activations(
        layer, quantize_activations)

  def get_output_quantizers(self, layer):
    outputs_and_quantizers = (
        self.config.get_output_quantizers(layer))
    return [self.fixed_quantizer
            for _ in outputs_and_quantizers]

  def get_config(self):
    return {
        'config': tf.keras.utils.serialize_keras_object(self.config),
        'num_bits': self.num_bits,
        'init_min': self.init_min,
        'init_max': self.init_max,
        'narrow_range': self.narrow_range}

  @classmethod
  def from_config(cls, config):
    config['config'] = tf.keras.utils.deserialize_keras_object(config['config'])
    return cls(**config)
