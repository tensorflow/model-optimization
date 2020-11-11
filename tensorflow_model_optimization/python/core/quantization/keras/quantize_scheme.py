# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Quantization scheme which specifies how quantization should be applied.

Module: tfmot.quantization.keras
"""

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class QuantizeScheme(object):
  """ABC interface which specifies transformer and quantization registry."""

  @abc.abstractmethod
  def get_layout_transformer(self):
    """Returns the layout transforms for this scheme.

    Returns:
      Returns the QuantizeLayoutTransform for this quantization scheme.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def get_quantize_registry(self):
    """Returns the quantization registry for this scheme.

    Returns:
      Returns the QuantizeRegistry for this quantization scheme.
    """
    raise NotImplementedError('Must be implemented in subclasses.')
