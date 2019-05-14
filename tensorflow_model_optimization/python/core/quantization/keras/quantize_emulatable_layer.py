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
"""Abstract Base Class for quantize emulation in custom keras layers."""

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class QuantizeEmulatableLayer(object):
  """Abstract Base Class for quantize emulation in custom keras layers.

  Custom keras layers which want to implement quantization of their operations
  during training should implement this class.

  """

  @abc.abstractmethod
  def get_quantizable_weights(self):
    """Returns list of quantizable weight tensors.

    All the weight tensors which the layer wants to be quantized during
    training must be returned by this method.

    Returns: List of weight tensors/kernels in the keras layer which must be
        quantized during training.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def set_quantizable_weights(self, weights):
    """Sets list of quantizable weight tensors.

    This method replaces the existing quantizable weight tensors for
    the layer with the specified set of weights.

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
    raise NotImplementedError('Must be implemented in subclasses.')
